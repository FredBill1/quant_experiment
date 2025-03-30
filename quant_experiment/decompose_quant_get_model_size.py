from multiprocessing import Pool

import pandas as pd
import torch

from .config import MODEL_PATH

SAVED_MODEL_DIR = MODEL_PATH.with_name("decompose_quant")
RESULTS_PATH = SAVED_MODEL_DIR / "tboard/results.csv"
RESULTS2_PATH = RESULTS_PATH.with_stem("results2")


def work(row) -> int:
    isna = row.isna()
    do_decompose = not isna["decompose_method"]
    do_finetune = row["do_finetune"] == "True"
    do_quant = not isna["quant_weight"]

    decompose_method, decompose_factor = (row["decompose_method"], row["decompose_factor"]) if do_decompose else (None, None)
    quant_weight, quant_act = (row["quant_weight"], row["quant_act"]) if do_quant else (None, None)

    prefix = f"{decompose_method}_{float(decompose_factor):g}" if do_decompose else "baseline"
    before_finetune_stem = f"{prefix}-before_finetune"
    after_finetune_stem = f"{prefix}-after_finetune"
    finetune_stem = after_finetune_stem if do_finetune else before_finetune_stem
    quantized_stem = f"{finetune_stem}-{quant_weight}-{quant_act}" if do_quant else finetune_stem
    file = SAVED_MODEL_DIR / f"{quantized_stem}.pth"
    if not (do_decompose or do_quant):
        file = MODEL_PATH
    ckpt = torch.load(file, map_location="cpu")
    if do_quant:
        ckpt = ckpt["state_dict"]
    model_size = sum(p.numel() * p.element_size() for p in ckpt.values())
    return model_size


def main() -> None:
    df = pd.read_csv(RESULTS_PATH, dtype=str)
    # for _, row in df.iterrows():
    #     work(row)
    with Pool() as pool:
        sizes = pool.map(work, [row for _, row in df.iterrows()])
    df["model_size"] = sizes
    df.to_csv(RESULTS2_PATH, index=False)


if __name__ == "__main__":
    main()
