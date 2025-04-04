from torchinfo import summary

from .config import IMAGE_SIZE, MODEL_NAME
from .models import create_model


def main() -> None:
    model = create_model(MODEL_NAME, quantable=False)
    summary(model, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE))


if __name__ == "__main__":
    main()
