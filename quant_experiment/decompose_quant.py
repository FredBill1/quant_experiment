import gc
from collections.abc import Generator
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import product
from pprint import pp
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from optimum.quanto import Calibration, freeze, qfloat8, qint2, qint4, qint8, quantize
from optuna.trial import FixedTrial
from rich.console import Console
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .methods.low_rank_decompose.decompose_model import Conv2dDecomposeMethod, decompose_model
from .models import create_model
from .utils.EarlyStopping import EarlyStopping
from .utils.quanto_save_load import quanto_load, quanto_save
from .utils.training import evaluate, get_device, train_one_epoch

console = Console()

FAST_MODE = False

SAVED_MODEL_DIR = MODEL_PATH.with_name("decompose_quant")
TBOARD_DIR = SAVED_MODEL_DIR / "tboard"

DECOMPOSE_METHODS = [Conv2dDecomposeMethod.TUCKER, Conv2dDecomposeMethod.CP]
DECOMPOSE_FACTORS = np.linspace(0.1, 0.9, 9).tolist()
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

DECOMPOSE_FINE_TUNE_MAX_EPOCHS = 200 if not FAST_MODE else 3
DATALOADER_CFG = dict(num_workers=6) if not FAST_MODE else dict(num_workers=0, persistent_workers=False)

if FAST_MODE:
    evaluate = partial(evaluate, max_step=5)
    train_one_epoch = partial(train_one_epoch, max_step=5)


@dataclass
class ExperimentConfig:
    decompose_method: Optional[Conv2dDecomposeMethod] = None
    decompose_factor: Optional[float] = None
    do_finetune: bool = False
    quant_weight: Optional[str] = None
    quant_act: Optional[str] = None

    def get_decompose_trail(self) -> FixedTrial:
        if self.decompose_method is not None:
            return FixedTrial({"decompose_rank_factor": self.decompose_factor, "decompose_method": self.decompose_method})
        return FixedTrial({})

    @property
    def do_decompose(self) -> bool:
        return self.decompose_method is not None

    @property
    def do_quant(self) -> bool:
        return self.quant_weight is not None

    @classmethod
    def all_configs(cls) -> Generator["ExperimentConfig", None, None]:
        def decompose_cfgs():
            yield (None, None, False)
            for d_method, d_factor, do_finetune in product(DECOMPOSE_METHODS, DECOMPOSE_FACTORS, (True, False)):
                yield (d_method, d_factor, do_finetune)

        def quant_cfgs():
            for w_type, a_type in product(QUANT_WEIGHT_DTYPES.keys(), QUANT_ACTIVATION_DTYPES.keys()):
                yield (w_type, a_type)

        yield cls(None, None, False, None, None)
        for decompose_cfg, quant_cfg in product(decompose_cfgs(), quant_cfgs()):
            yield cls(*decompose_cfg, *quant_cfg)


@dataclass
class ExperimentVars:
    device: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def model_size(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.state_dict().values())


class Experiment:
    def __init__(
        self,
        cfg: ExperimentConfig,
        vars: ExperimentVars,
    ) -> None:
        self.cfg = cfg
        self.vars = vars

        self.criterion = nn.CrossEntropyLoss()

        self.prefix = f"{cfg.decompose_method}_{cfg.decompose_factor:g}" if cfg.do_decompose else "baseline"
        before_finetune_stem = f"{self.prefix}-before_finetune"
        after_finetune_stem = f"{self.prefix}-after_finetune"
        quantized_stem = f"{after_finetune_stem if cfg.do_finetune else before_finetune_stem}-{cfg.quant_weight}-{cfg.quant_act}"
        self.before_finetune_path = SAVED_MODEL_DIR / f"{before_finetune_stem}.pth"
        self.after_finetune_path = SAVED_MODEL_DIR / f"{after_finetune_stem}.pth"
        self.quantized_path = SAVED_MODEL_DIR / f"{quantized_stem}.pth"
        self.results_path = TBOARD_DIR / f"results.csv"

    @staticmethod
    def write_metrics(
        writer: SummaryWriter,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        test_loss: float,
        test_acc: float,
        epoch: Optional[int],
    ) -> None:
        writer.add_scalar(f"loss/train", train_loss, epoch)
        writer.add_scalar(f"loss/val", val_loss, epoch)
        writer.add_scalar(f"loss/test", test_loss, epoch)
        writer.add_scalar(f"acc/train", train_acc, epoch)
        writer.add_scalar(f"acc/val", val_acc, epoch)
        writer.add_scalar(f"acc/test", test_acc, epoch)
        console.print(
            f"[yellow]{train_loss=:.4f}[/yellow] [green]{train_acc=:.4f}[/green] "
            f"[yellow]{val_loss=:.4f}[/yellow] [green]{val_acc=:.4f}[/green] "
            f"[yellow]{test_loss=:.4f}[/yellow] [green]{test_acc=:.4f}[/green]"
        )

    def get_decomposed_model_without_finetune(self) -> nn.Module:
        console.print("[bold cyan]`get_decomposed_model_without_finetune`[/bold cyan]")
        model = create_model(MODEL_NAME, quantable=True).to(self.vars.device)

        if not self.cfg.do_decompose:
            console.print("[yellow]Not decomposing model[/yellow]")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=self.vars.device))
            return model

        if self.before_finetune_path.exists():
            console.print("[green]Decomposed model exists, loading decomposed model[/green]")
            decompose_model(model, self.cfg.get_decompose_trail(), do_calculation=False, layerwise=False, skip_linear=True, verbose=False)
            model.load_state_dict(torch.load(self.before_finetune_path, map_location=self.vars.device))
            return model

        console.print("[bold blue]Decomposing model[/bold blue]")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.vars.device))
        decompose_model(model, self.cfg.get_decompose_trail(), do_calculation=True, layerwise=False, skip_linear=True)
        torch.save(model.state_dict(), self.before_finetune_path)
        return model

    def write_results(self, data: dict) -> None:
        df = pd.DataFrame([data])
        df.to_csv(self.results_path, mode="a", index=False, header=not self.results_path.exists())
        console.print(f"[bold green]Results saved to {self.results_path}[/bold green]:")
        pp(data)

    def get_decomposed_finetuned_model(self) -> nn.Module:
        console.print("[bold cyan]`get_decomposed_finetuned_model`[/bold cyan]")
        model = self.get_decomposed_model_without_finetune()

        if not self.cfg.do_decompose or not self.cfg.do_finetune:
            console.print("[yellow]Not finetuning model[/yellow]")
            return model

        if self.after_finetune_path.exists():
            console.print("[green]Finetuned model exists, loading finetuned model[/green]")
            model.load_state_dict(torch.load(self.after_finetune_path, map_location=self.vars.device))
            return model

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        scheduler = EarlyStopping.create_lr_scheduler(optimizer)
        early_stopping = EarlyStopping()
        scaler = torch.amp.GradScaler()

        with SummaryWriter(str(TBOARD_DIR / self.prefix)) as writer:
            console.print("[bold blue]Evaluating model before finetune[/bold blue]")
            train_loss, train_acc = evaluate(model, self.vars.train_loader, self.criterion, self.vars.device, desc_prefix="Train: ")
            val_loss, val_acc = evaluate(model, self.vars.val_loader, self.criterion, self.vars.device, desc_prefix="Val:   ")
            test_loss, test_acc = evaluate(model, self.vars.test_loader, self.criterion, self.vars.device, desc_prefix="Test:  ")
            self.write_metrics(writer, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, 0)

            data = asdict(replace(self.cfg, do_finetune=False, quant_weight=None, quant_act=None))
            results = dict(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                model_size=model_size(model),
            )
            data.update(results)
            self.write_results(data)

            for epoch in range(1, DECOMPOSE_FINE_TUNE_MAX_EPOCHS + 1):
                console.print(f"[bold magenta]Epoch {epoch}[/bold magenta]")
                train_loss, train_acc = train_one_epoch(model, self.vars.train_loader, self.criterion, optimizer, self.vars.device, scaler=scaler)
                val_loss, val_acc = evaluate(model, self.vars.val_loader, self.criterion, self.vars.device, desc_prefix="Val:   ")
                test_loss, test_acc = evaluate(model, self.vars.test_loader, self.criterion, self.vars.device, desc_prefix="Test:  ")
                self.write_metrics(writer, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, epoch)

                if early_stopping(val_loss, model):
                    console.print("[red]Early stopping[/red]")
                    break

                scheduler.step(val_loss)
                console.print(f"[yellow]Learning rate: {scheduler.get_last_lr()}[/yellow]")

            console.print("[bold blue]Evaluating model after finetune[/bold blue]")
            model.load_state_dict(early_stopping.best_state_dict)
            train_loss, train_acc = evaluate(model, self.vars.train_loader, self.criterion, self.vars.device, desc_prefix="Train: ")
            val_loss, val_acc = evaluate(model, self.vars.val_loader, self.criterion, self.vars.device, desc_prefix="Val:   ")
            test_loss, test_acc = evaluate(model, self.vars.test_loader, self.criterion, self.vars.device, desc_prefix="Test:  ")

            data = asdict(replace(self.cfg, do_finetune=True, quant_weight=None, quant_act=None))
            results = dict(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                model_size=model_size(model),
            )
            data.update(results)
            self.write_results(data)

        torch.save(model.state_dict(), self.after_finetune_path)
        return model

    def get_quantized_model(self) -> nn.Module:
        console.print("[bold cyan]`get_quantized_model`[/bold cyan]")
        model = self.get_decomposed_finetuned_model()

        if not self.cfg.do_quant:
            console.print("[yellow]Not quantizing model[/yellow]")
            return model

        if self.quantized_path.exists():
            console.print("[green]Quantized model exists, loading quantized model[/green]")
            return quanto_load(model, self.quantized_path, self.vars.device)

        console.print("[bold blue]Quantizing model[/bold blue]")
        quantize(model, QUANT_WEIGHT_DTYPES[self.cfg.quant_weight], QUANT_ACTIVATION_DTYPES[self.cfg.quant_act])
        console.print("[bold blue]Calibrating model[/bold blue]")
        with Calibration(streamline=False):
            evaluate(model, self.vars.train_loader, self.criterion, self.vars.device, desc_prefix="Calib: ")
        freeze(model)
        quanto_save(model, self.quantized_path)

        # recreate the model to avoid the issue https://github.com/huggingface/optimum-quanto/issues/378
        console.print("[yellow]Recreating quantized model[/yellow]")
        return self.get_quantized_model()

    def run(self) -> None:
        data = asdict(self.cfg)
        pp(data)
        if self.quantized_path.exists() or (not self.cfg.do_decompose and not self.cfg.do_quant and self.results_path.exists()):
            console.print("[green]Quantized model exists, skipping experiment[/green]")
            return
        console.print("[bold cyan]===== Running experiment =====[/bold cyan]")
        model = self.get_quantized_model()
        console.print("[bold magenta]Evaluating final model[/bold magenta]")
        train_loss, train_acc = evaluate(model, self.vars.train_loader, self.criterion, self.vars.device, desc_prefix="Train: ")
        val_loss, val_acc = evaluate(model, self.vars.val_loader, self.criterion, self.vars.device, desc_prefix="Val:   ")
        test_loss, test_acc = evaluate(model, self.vars.test_loader, self.criterion, self.vars.device, desc_prefix="Test:  ")

        results = dict(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            model_size=model_size(model),
        )
        data.update(results)
        self.write_results(data)


def main() -> None:
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TBOARD_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    train_loader = get_imagewoof_dataloader(DatasetSplit.TRAIN, **DATALOADER_CFG)
    val_loader = get_imagewoof_dataloader(DatasetSplit.VAL, **DATALOADER_CFG)
    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, **DATALOADER_CFG)

    exp_vars = ExperimentVars(device=device, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    for i, exp_cfg in enumerate(all_configs := list(ExperimentConfig.all_configs()), 1):
        console.rule(f"[bold cyan]Running experiment {i}/{len(all_configs)}[/bold cyan]")
        Experiment(exp_cfg, exp_vars).run()

        console.print("[bold blue]Clearing cache[/bold blue]")
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
