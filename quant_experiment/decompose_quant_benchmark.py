import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from optuna.trial import FixedTrial
from tqdm import tqdm

from .config import MODEL_NAME, MODEL_PATH
from .methods.low_rank_decompose.decompose_model import decompose_model
from .models import create_model
from .utils.quanto_save_load import quanto_load
from .utils.training import get_device

SAVED_MODEL_DIR = MODEL_PATH.with_name("decompose_quant")
RESULTS_PATH = SAVED_MODEL_DIR / "tboard/results.csv"

DECOMPOSE_FACTORS = np.linspace(0.1, 0.9, 9).tolist()


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

    if do_quant:
        model = quanto_load(model, file, device)
    else:
        model.load_state_dict(torch.load(file, map_location=device))

    return model


def main() -> None:
    device = get_device()

    df = pd.read_csv(RESULTS_PATH)
    for id, row in tqdm(df.iterrows(), total=len(df), desc="Loading models"):
        load_model(row, device)
    print("Models loaded successfully.")


if __name__ == "__main__":
    main()
