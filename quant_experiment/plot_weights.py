import matplotlib.pyplot as plt
import numpy as np
import torch
from optuna.trial import FixedTrial
from tqdm import tqdm

from .config import MODEL_NAME, MODEL_PATH
from .methods.low_rank_decompose.decompose_model import decompose_model
from .models import create_model


def main() -> None:
    model = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True)
    decompose_model(
        model,
        FixedTrial(
            {
                "decompose_rank_factor": np.float32(0.3),
                "decompose_method": "cp",
            }
        ),
        do_calculation=False,
        layerwise=False,
        skip_linear=True,
    )
    ckpt_path = MODEL_PATH.parent / "search/cp_0.3.pth"
    model.load_state_dict(torch.load(ckpt_path))

    params = list(model.named_parameters())

    num_cols = 8
    num_rows = len(params) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
    for ax, (name, param) in zip(axs.flat, tqdm(params)):
        ax.hist(param.detach().cpu().numpy().flatten(), bins=100)
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(MODEL_PATH.with_name(MODEL_PATH.stem + "_params.png"))


if __name__ == "__main__":
    main()
