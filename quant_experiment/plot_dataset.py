import matplotlib.pyplot as plt
import numpy as np

from .config import SEED
from .data.common import tensor_to_numpy_image
from .data.imagewoof import DatasetSplit, get_imagewoof_dataset


def main() -> None:
    dev_dataset, classes = get_imagewoof_dataset(DatasetSplit.TRAIN_AND_VAL)
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    ids = np.random.default_rng(SEED).choice(len(dev_dataset), 10, replace=False)

    for i, ax in zip(ids, axes.flatten()):
        sample, target = dev_dataset[i]
        sample = tensor_to_numpy_image(sample)
        ax.imshow(sample)
        ax.set_title(classes[target])
        ax.axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
