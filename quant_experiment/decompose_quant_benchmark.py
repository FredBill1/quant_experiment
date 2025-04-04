from itertools import islice
from time import perf_counter_ns

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from optuna.trial import FixedTrial
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .methods.low_rank_decompose.decompose_model import decompose_model
from .models import create_model
from .utils.quanto_save_load import quanto_load
from .utils.training import get_device

SAVED_MODEL_DIR = MODEL_PATH.with_name("decompose_quant")
RESULTS_PATH = SAVED_MODEL_DIR / "tboard/results.csv"
BENCHMARK_RESULTS_PATH = RESULTS_PATH.with_stem("benchmark_results")
DECOMPOSE_FACTORS = np.linspace(0.1, 0.9, 9).tolist()


def run_one_epoch(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    total_time_ns = 0
    total_samples = 0
    for inputs, _ in tqdm(dataloader):
        inputs = inputs.to(device)
        with torch.no_grad():
            start = perf_counter_ns()
            model(inputs)
            end = perf_counter_ns()
            total_time_ns += end - start
            total_samples += inputs.size(0)
    return total_time_ns / max(1, total_samples)


def load_model(row: pd.Series, device: str) -> nn.Module:
    isna = row.isna()
    do_decompose = not isna["decompose_method"]
    do_finetune = row["do_finetune"] == "True"
    do_quant = not isna["quant_weight"]

    model = create_model(MODEL_NAME, quantable=True).to(device)
    if not (do_decompose or do_quant):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        return model

    decompose_method, decompose_factor = (row["decompose_method"], row["decompose_factor"]) if do_decompose else (None, None)
    quant_weight, quant_act = (row["quant_weight"], row["quant_act"]) if do_quant else (None, None)

    prefix = f"{decompose_method}_{float(decompose_factor):g}" if do_decompose else "baseline"
    before_finetune_stem = f"{prefix}-before_finetune"
    after_finetune_stem = f"{prefix}-after_finetune"
    finetune_stem = after_finetune_stem if do_finetune else before_finetune_stem
    quantized_stem = f"{finetune_stem}-{quant_weight}-{quant_act}" if do_quant else finetune_stem
    file = SAVED_MODEL_DIR / f"{quantized_stem}.pth"

    if do_decompose:
        factor = DECOMPOSE_FACTORS[np.argmin([abs(float(decompose_factor) - x) for x in DECOMPOSE_FACTORS])]
        decompose_config = {
            "decompose_rank_factor": factor,
            "decompose_method": decompose_method,
        }
        decompose_model(model, FixedTrial(decompose_config), do_calculation=False, layerwise=False, skip_linear=True, verbose=False)
        model.to(device)

    if do_quant:
        model = quanto_load(model, file, device)
    else:
        model.load_state_dict(torch.load(file, map_location=device))

    return model


def main() -> None:
    device = get_device()
    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=6)
    df = pd.read_csv(RESULTS_PATH)

    finished = len(pd.read_csv(BENCHMARK_RESULTS_PATH)) if BENCHMARK_RESULTS_PATH.exists() else 0

    for _, row in tqdm(islice(df.iterrows(), finished, len(df)), total=len(df) - finished, desc="Loading models"):
        model = load_model(row, device)
        time_per_image_ns = run_one_epoch(model, test_loader, device)
        df_row = pd.DataFrame([row])
        df_row["time_per_image_ns"] = time_per_image_ns
        df_row.to_csv(BENCHMARK_RESULTS_PATH, mode="a", header=not BENCHMARK_RESULTS_PATH.exists(), index=False)


if __name__ == "__main__":
    main()
