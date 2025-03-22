from typing import TYPE_CHECKING

import torch
from torch.utils.tensorboard import SummaryWriter

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .models import create_model
from .utils.EarlyStopping import EarlyStopping
from .utils.training import evaluate, get_device, train_one_epoch

if TYPE_CHECKING:
    import torch.ao.quantization
    import torch.nn.intrinsic.qat


QAT_MIN_EPOCHS = 10
QAT_MAX_EPOCHS = 200


def main():
    device = get_device()

    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=2)
    train_loader = get_imagewoof_dataloader(DatasetSplit.TRAIN, num_workers=6)
    val_loader = get_imagewoof_dataloader(DatasetSplit.VAL, num_workers=6)
    criterion = torch.nn.CrossEntropyLoss()

    def dynamic() -> None:  #! Note that Conv2d does not support dynamic quantization yet
        print("Dynamic quantization")
        model_fp32 = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True, quantize=False)
        model_fp32.load_state_dict(torch.load(MODEL_PATH))
        quant_device = "cpu"  # `does not support GPU yet
        model_fp32.to(quant_device)

        model_int8 = torch.ao.quantization.quantize_dynamic(
            model_fp32,
            qconfig_spec={
                torch.nn.Linear: torch.ao.quantization.default_dynamic_qconfig,
                # torch.nn.Conv2d: torch.ao.quantization.default_dynamic_qconfig,  # not supported
            },
            dtype=torch.qint8,
        )
        torch.save(model_int8.state_dict(), MODEL_PATH.with_stem(MODEL_PATH.stem + "_torch_dynamic"))

        test_loss, test_acc = evaluate(model_int8, test_loader, criterion, quant_device)
        print(f"{test_loss=} {test_acc=}")

    def static() -> None:
        print("Static quantization")
        model_fp32 = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True, quantize=False)
        model_fp32.load_state_dict(torch.load(MODEL_PATH))
        model_fp32.to(device)

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
        quant_device = "cpu"  # `qscheme: per_channel_affine` does not support GPU yet

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
        print("Calibrating...")
        evaluate(model_fp32_prepared, train_loader, criterion, device)

        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, and replaces key operators with quantized
        # implementations.
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared.to(quant_device))
        torch.save(model_int8.state_dict(), MODEL_PATH.with_stem(MODEL_PATH.stem + "_torch_static"))

        print("Evaluated quantized model:")
        test_loss, test_acc = evaluate(model_int8, test_loader, criterion, quant_device)
        print(f"{test_loss=} {test_acc=}")

    def qat() -> None:
        print("Quantization Aware Training")
        model_fp32 = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True, quantize=False)
        model_fp32.load_state_dict(torch.load(MODEL_PATH))
        model_fp32.to(device)

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
        quant_device = "cpu"  # `qscheme: per_channel_affine` does not support GPU yet

        # fuse the activations to preceding layers, where applicable
        # this needs to be done manually depending on the model architecture
        model_fp32.fuse_model(is_qat=True)
        model_fp32_fused = model_fp32

        # Prepare the model for QAT. This inserts observers and fake_quants in
        # the model needs to be set to train for QAT logic to work
        # the model that will observe weight and activation tensors during calibration.
        model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused.train())

        # run the training loop (not shown)
        with SummaryWriter(log_dir=MODEL_PATH.parent / "qat") as writer:
            optimizer = torch.optim.SGD(model_fp32_prepared.parameters(), lr=1e-4, momentum=0.9)
            scheduler = EarlyStopping.create_lr_scheduler(optimizer)
            early_stopping = EarlyStopping()
            for epoch in range(1, QAT_MAX_EPOCHS + 1):
                print(f"Epoch {epoch}")
                if epoch >= 3:
                    model_fp32_prepared.apply(torch.ao.quantization.disable_observer)
                if epoch >= 2:
                    model_fp32_prepared.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                train_loss, train_acc = train_one_epoch(model_fp32_prepared, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model_fp32_prepared, val_loader, criterion, device)
                writer.add_scalar("train/loss", train_loss, epoch)
                writer.add_scalar("train/accuracy", train_acc, epoch)
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/accuracy", val_acc, epoch)
                if early_stopping(val_loss, model_fp32_prepared) and epoch >= QAT_MIN_EPOCHS:
                    print("Early stopping")
                    break
                scheduler.step(val_loss)
                print(f"Learning rate: {scheduler.get_last_lr()}")

        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, fuses modules where appropriate,
        # and replaces key operators with quantized implementations.
        model_fp32_prepared.eval()
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared.to(quant_device))
        torch.save(model_int8.state_dict(), MODEL_PATH.with_stem(MODEL_PATH.stem + "_torch_qat"))

        # run the model, relevant calculations will happen in int8
        test_loss, test_acc = evaluate(model_int8, test_loader, criterion, quant_device)
        print(f"{test_loss=} {test_acc=}")

    def test_static():
        model = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True, quantize=False)

        model.qconfig = torch.ao.quantization.get_default_qconfig("x86")
        quant_device = "cpu"
        model.eval()
        model.fuse_model(is_qat=False)
        torch.ao.quantization.prepare(model, inplace=True)
        torch.ao.quantization.convert(model, inplace=True)

        model.load_state_dict(torch.load(MODEL_PATH.with_stem(MODEL_PATH.stem + "_torch_static")))
        model.to(quant_device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, quant_device)
        print(f"{test_loss=} {test_acc=}")

    def test_qat():
        model = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True, quantize=False)

        model.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        quant_device = "cpu"
        model.eval()
        model.fuse_model(is_qat=False)
        torch.ao.quantization.prepare(model, inplace=True)
        torch.ao.quantization.convert(model, inplace=True)

        model.load_state_dict(torch.load(MODEL_PATH.with_stem(MODEL_PATH.stem + "_torch_qat")))
        model.to(quant_device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, quant_device)
        print(f"{test_loss=} {test_acc=}")

    dynamic()
    # static()
    # qat()
    # test_static()
    # test_qat()


if __name__ == "__main__":
    main()
