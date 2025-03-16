from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from .common import IMAGE_SIZE, IMAGEWOOF_DIRS, DatasetSplit

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATA_TRANSFORMS = {
    DatasetSplit.TRAIN: T.Compose(
        [
            T.RandomResizedCrop(IMAGE_SIZE),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    ),
    DatasetSplit.VAL: T.Compose(
        [
            T.Resize(round(256 / 224 * IMAGE_SIZE)),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    ),
}


def get_imagewoof_dataset(split: DatasetSplit):
    return ImageFolder(
        root=IMAGEWOOF_DIRS[split],
        transform=DATA_TRANSFORMS[split],
    )


if __name__ == "__main__":
    import PIL.Image
    import torch

    train_dataset = get_imagewoof_dataset(DatasetSplit.TRAIN)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    img: torch.Tensor = train_dataset[0][0]
    print(f"Sample image shape: {img.shape}")
    img = img.permute(1, 2, 0)  # C, H, W -> H, W, C
    img = img * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN)
    PIL.Image.fromarray((img * 255).byte().numpy()).show()
