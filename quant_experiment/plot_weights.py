import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from .config import MODEL_NAME, MODEL_PATH
from .models import create_model


def main() -> None:
    model = create_model(MODEL_NAME, from_pretrained=False, frozen=False)
    model.load_state_dict(torch.load(MODEL_PATH))

    params = list(model.named_parameters())

    num_cols = 8
    num_rows = len(params) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
    for ax, (name, param) in zip(axs.flat, tqdm(params)):
        ax.hist(param.detach().cpu().numpy().flatten())
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(MODEL_PATH.with_name(MODEL_PATH.stem + "_params.png"))


if __name__ == "__main__":
    main()
