"https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html"

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .models import create_model
from .utils.training import evaluate, get_device


def main() -> None:
    device = get_device()

    model = create_model(MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)

    train_loader = get_imagewoof_dataloader(DatasetSplit.TRAIN, num_workers=0, persistent_workers=False)
    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=0, persistent_workers=False)

    # test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    # print(f"{test_loss=}, {test_acc=}")

    def forward_loop(model: nn.Module) -> None:
        for batch in tqdm(train_loader):
            model(batch[0].to(device))

    config = mtq.INT8_SMOOTHQUANT_CFG

    model = mtq.quantize(model, config, forward_loop)
    mtq.print_quant_summary(model)
    mto.save(model, MODEL_PATH.with_stem("model_quant_trt"))
    mtq.fold_weight(model)
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"{test_loss=}, {test_acc=}")


if __name__ == "__main__":
    main()
