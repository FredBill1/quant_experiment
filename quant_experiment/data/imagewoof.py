import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder

from ..config import DATALOADER_ARGS, IMAGEWOOF_TEST_DIR, IMAGEWOOF_TRAIN_DIR, SEED, DatasetSplit
from .common import TRAIN_TRANSFORMS, VAL_TRANSFORMS, tensor_to_numpy_image


def get_imagewoof_dataset(split: DatasetSplit, random_state: int = SEED) -> tuple[Dataset[tuple[torch.Tensor, int]], list[str]]:
    transforms = TRAIN_TRANSFORMS if split == DatasetSplit.TRAIN else VAL_TRANSFORMS
    if split == DatasetSplit.TEST:
        dataset = ImageFolder(IMAGEWOOF_TEST_DIR, transform=transforms)
        return dataset, dataset.classes
    dataset = ImageFolder(IMAGEWOOF_TRAIN_DIR, transform=transforms)
    if split == DatasetSplit.TRAIN_AND_VAL:
        return dataset, dataset.classes
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, shuffle=True, stratify=dataset.targets, random_state=random_state)
    return Subset(dataset, train_idx if split == DatasetSplit.TRAIN else val_idx), dataset.classes


def get_imagewoof_dataloader(
    split: DatasetSplit,
    *,
    random_state: int = SEED,
    **kwargs,
) -> torch.utils.data.DataLoader:
    dataset, _ = get_imagewoof_dataset(split, random_state)
    dataloader_kwargs = DATALOADER_ARGS.copy()
    dataloader_kwargs.update(kwargs)
    if split == DatasetSplit.TRAIN:
        dataloader_kwargs["shuffle"] = True
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


if __name__ == "__main__":
    import PIL.Image

    train_dataset, classes = get_imagewoof_dataset(DatasetSplit.TRAIN)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Classes: {classes}")
    sample, target = train_dataset[0]
    print(f"Sample shape: {sample.shape}, target: {target}")
    sample = tensor_to_numpy_image(sample)
    PIL.Image.fromarray(sample).show()
