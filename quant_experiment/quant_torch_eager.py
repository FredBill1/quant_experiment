from typing import TYPE_CHECKING

import torch

from .config import DATALOADER_ARGS
from .data.imagewoof import DatasetSplit, get_imagewoof_dataset
from .models.resnet18 import create_model
from .utils.training import train_one_epoch, val_one_epoch

if TYPE_CHECKING:
    import torch.ao.quantization

DEVICE = "cpu"  # PyTorch latest version 2.6 does not support quantization on CUDA yet

FINETUNE_EPOCH = 5
FINETUNE_LR = 1e-5


def main():
    train_data = get_imagewoof_dataset(DatasetSplit.TRAIN)[0]
    train_loader = torch.utils.data.DataLoader(train_data, **DATALOADER_ARGS)
    val_data = get_imagewoof_dataset(DatasetSplit.VAL)[0]
    val_loader = torch.utils.data.DataLoader(val_data, **DATALOADER_ARGS)
    test_data = get_imagewoof_dataset(DatasetSplit.TEST)[0]
    test_loader = torch.utils.data.DataLoader(test_data, **DATALOADER_ARGS)
    criterion = torch.nn.CrossEntropyLoss()

    def dynamic() -> None:  #! Note that Conv2d does not support dynamic quantization yet
        print("Dynamic quantization")
        model_fp32 = create_model(from_pretrained=False, frozen=False, quantable=False, quantize=False)
        model_fp32.load_state_dict(torch.load("runs/Mar16_23-43-58_FredBill/model.pth"))

        model_int8 = torch.ao.quantization.quantize_dynamic(
            model_fp32,
            qconfig_spec=None,
            dtype=torch.quint4x2,
        )
        model_int8.to(DEVICE)

        test_loss, test_acc = val_one_epoch(model_int8, test_loader, criterion, DEVICE)
        print(f"{test_loss=} {test_acc=}")

    def static() -> None:
        print("Static quantization")
        model_fp32 = create_model(from_pretrained=False, frozen=False, quantable=True, quantize=False)
        model_fp32.load_state_dict(torch.load("runs/Mar16_23-43-58_FredBill/model.pth"))

        # model must be set to eval mode for static quantization logic to work
        model_fp32.eval()

        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'x86' for server inference and 'qnnpack'
        # for mobile inference. Other quantization configurations such as selecting
        # symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
        # can be specified here.
        # Note: the old 'fbgemm' is still available but 'x86' is the recommended default
        # for server inference.
        # model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("x86")  # TODO: can be changed

        # Fuse the activations to preceding layers, where applicable.
        # This needs to be done manually depending on the model architecture.
        # Common fusions include `conv + relu` and `conv + batchnorm + relu`
        model_fp32.fuse_model(is_qat=False)
        model_fp32_fused = model_fp32

        # Prepare the model for static quantization. This inserts observers in
        # the model that will observe activation tensors during calibration.
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)

        # calibrate the prepared model to determine quantization parameters for activations
        # in a real world setting, the calibration would be done with a representative dataset
        val_one_epoch(model_fp32_prepared, train_loader, criterion, DEVICE)

        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, and replaces key operators with quantized
        # implementations.
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

        test_loss, test_acc = val_one_epoch(model_int8, test_loader, criterion, DEVICE)
        print(f"{test_loss=} {test_acc=}")

    def qat() -> None:
        print("Quantization Aware Training")
        model_fp32 = create_model(from_pretrained=False, frozen=False, quantable=True, quantize=False)
        model_fp32.load_state_dict(torch.load("runs/Mar16_23-43-58_FredBill/model.pth"))

        # model must be set to train mode for QAT logic to work
        model_fp32.train()

        # model must be set to eval for fusion to work
        model_fp32.eval()

        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'x86' for server inference and 'qnnpack'
        # for mobile inference. Other quantization configurations such as selecting
        # symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
        # can be specified here.
        # Note: the old 'fbgemm' is still available but 'x86' is the recommended default
        # for server inference.
        # model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")  # TODO: can be changed

        # fuse the activations to preceding layers, where applicable
        # this needs to be done manually depending on the model architecture
        model_fp32.fuse_model(is_qat=False)
        model_fp32_fused = model_fp32

        # Prepare the model for QAT. This inserts observers and fake_quants in
        # the model needs to be set to train for QAT logic to work
        # the model that will observe weight and activation tensors during calibration.
        model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused.train())

        # run the training loop (not shown)
        optimizer = torch.optim.Adam(model_fp32_prepared.parameters(), lr=FINETUNE_LR)
        for epoch in range(1, FINETUNE_EPOCH + 1):
            print(f"Epoch {epoch}")
            train_one_epoch(model_fp32_prepared, train_loader, criterion, optimizer, DEVICE)
            val_one_epoch(model_fp32_prepared, val_loader, criterion, DEVICE)

        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, fuses modules where appropriate,
        # and replaces key operators with quantized implementations.
        model_fp32_prepared.eval()
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

        # run the model, relevant calculations will happen in int8
        test_loss, test_acc = val_one_epoch(model_int8, test_loader, criterion, DEVICE)
        print(f"{test_loss=} {test_acc=}")

    # dynamic()
    static()
    # qat()


if __name__ == "__main__":
    main()
