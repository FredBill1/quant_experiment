import json
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from optimum.quanto import Calibration, freeze, qfloat8, qint2, qint4, qint8, quantization_map, quantize, requantize
from optuna.trial import FixedTrial
from torch.utils.tensorboard import SummaryWriter

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .methods.low_rank_decompose.decompose_model import Conv2dDecomposeMethod, decompose_model
from .models import create_model
from .utils.EarlyStopping import EarlyStopping
from .utils.training import evaluate, get_device, train_one_epoch

LOG_DIR = MODEL_PATH.with_name("search")
TBOARD_DIR = LOG_DIR / "tboard"

DECOMPOSE_METHODS = [None, Conv2dDecomposeMethod.TUCKER, Conv2dDecomposeMethod.CP]
DECOMPOSE_RATIOS = np.linspace(0.1, 0.9, 9)
QUANT_WEIGHT_DTYPES = {
    "int2": qint2,
    "int4": qint4,
    "int8": qint8,
    "float8": qfloat8,
}
QUANT_ACTIVATION_DTYPES = {
    "int8": qint8,
    "float8": qfloat8,
    "dynamic": None,
}

QUANT_CONFIGS = list(product(QUANT_WEIGHT_DTYPES.keys(), QUANT_ACTIVATION_DTYPES.keys()))


DECOMPOSE_FINE_TUNE_MAX_EPOCHS = 200


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True, parents=True)

    device = get_device()

    train_loader = get_imagewoof_dataloader(DatasetSplit.TRAIN, num_workers=6)
    val_loader = get_imagewoof_dataloader(DatasetSplit.VAL, num_workers=6)
    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=6)
    criterion = nn.CrossEntropyLoss()

    for d_method in DECOMPOSE_METHODS:
        for d_ratio in DECOMPOSE_RATIOS if d_method is not None else [None]:
            file_prefix = f"{d_method}_{d_ratio:g}" if d_method is not None else "baseline"
            decomposed_model_path = LOG_DIR / f"{file_prefix}.pth"
            if decomposed_model_path.exists() and all((LOG_DIR / f"{file_prefix}_{w}_{a}.pth").exists() for w, a in QUANT_CONFIGS):
                continue
            model = create_model(MODEL_NAME, quantable=True)
            decompose_config = {
                "decompose_rank_factor": d_ratio,
                "decompose_method": d_method,
            }

            if decomposed_model_path.exists():
                model.to(device)
                decompose_model(model, FixedTrial(decompose_config), do_calculation=False, layerwise=False, skip_linear=True)
                model.load_state_dict(torch.load(decomposed_model_path))
            else:
                print(decomposed_model_path)
                model.load_state_dict(torch.load(MODEL_PATH))
                model.to(device)
                if d_method is not None:
                    decompose_model(model, FixedTrial(decompose_config), do_calculation=True, layerwise=False, skip_linear=True)
                    test_loss_before_finetune, test_acc_before_finetune = evaluate(model, test_loader, criterion, device)
                    print(f"{test_loss_before_finetune=} {test_acc_before_finetune=}")

                    with SummaryWriter(str(TBOARD_DIR / f"{file_prefix}")) as writer:

                        def add_scalars(train_loss, train_acc, val_loss, val_acc, epoch):
                            writer.add_scalar("train/loss", train_loss, epoch)
                            writer.add_scalar("train/accuracy", train_acc, epoch)
                            writer.add_scalar("val/loss", val_loss, epoch)
                            writer.add_scalar("val/accuracy", val_acc, epoch)
                            print(f"{train_loss=} {train_acc=} {val_loss=} {val_acc=}")

                        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
                        scheduler = EarlyStopping.create_lr_scheduler(optimizer)
                        early_stopping = EarlyStopping()
                        scaler = torch.amp.GradScaler()

                        train_loss_before_finetune, train_acc_before_finetune = evaluate(model, train_loader, criterion, device)
                        val_loss_before_finetune, val_acc_before_finetune = evaluate(model, val_loader, criterion, device)
                        state_dict_before_finetune = deepcopy(model.state_dict())
                        add_scalars(train_loss_before_finetune, train_acc_before_finetune, val_loss_before_finetune, val_acc_before_finetune, 0)

                        for epoch in range(1, DECOMPOSE_FINE_TUNE_MAX_EPOCHS + 1):
                            print(f"Epoch {epoch}")
                            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
                            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                            add_scalars(train_loss, train_acc, val_loss, val_acc, epoch)
                            if early_stopping(val_loss, model):
                                print("Early stopping")
                                break
                            scheduler.step(val_loss)
                            print(f"Learning rate: {scheduler.get_last_lr()}")

                        model.load_state_dict(
                            early_stopping.best_state_dict if early_stopping.best_loss < val_loss_before_finetune else state_dict_before_finetune
                        )

                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                if d_method is not None:
                    df = pd.DataFrame({"test_loss": [test_loss_before_finetune, test_loss], "test_acc": [test_acc_before_finetune, test_acc]})
                else:
                    df = pd.DataFrame({"test_loss": [test_loss], "test_acc": [test_acc]})
                df.to_csv(decomposed_model_path.with_suffix(".csv"), index=False)
                print(f"{test_loss=} {test_acc=}")
                torch.save(model.state_dict(), decomposed_model_path)

            for w_dtype, a_dtype in QUANT_CONFIGS:
                quantized_model_path = LOG_DIR / f"{file_prefix}_{w_dtype}_{a_dtype}.pth"
                if quantized_model_path.exists():
                    continue
                print(quantized_model_path)
                quantized_model = deepcopy(model)
                quantized_model.to(device)
                quantize(quantized_model, weights=QUANT_WEIGHT_DTYPES[w_dtype], activations=QUANT_ACTIVATION_DTYPES[a_dtype])
                with Calibration():
                    evaluate(quantized_model, train_loader, criterion, device)
                freeze(quantized_model)

                # https://github.com/huggingface/optimum-quanto/issues/378
                # recreate the model to avoid the issue
                quant_state_dict = quantized_model.state_dict()
                quant_map = quantization_map(quantized_model)
                quantized_model = deepcopy(model)
                requantize(quantized_model, quant_state_dict, quant_map, device)

                test_loss, test_acc = evaluate(quantized_model, test_loader, criterion, device)
                pd.DataFrame({"test_loss": [test_loss], "test_acc": [test_acc]}).to_csv(quantized_model_path.with_suffix(".csv"), index=False)
                print(f"{test_loss=} {test_acc=}")
                torch.save(quantized_model.state_dict(), quantized_model_path)
                with quantized_model_path.with_suffix(".json").open("w") as f:
                    json.dump(quant_map, f)


if __name__ == "__main__":
    main()
