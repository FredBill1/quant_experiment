# Quant Experiment

[optimum-quanto](https://github.com/huggingface/optimum-quanto), [Reference Code](https://github.com/huggingface/optimum-quanto/blob/main/examples/vision/image-classification/pets/quantize_vit_model.py).

[PyTorch Eager Mode Quantization](https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization), [Reference Code](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html).

## Get Started

### 1. Prepare Dataset

Dataset: [Imagewoof](https://github.com/fastai/imagenette?tab=readme-ov-file#imagewoof)

Unzip the dataset to `datasets/imagewoof2-160`:

```
📂datasets
 ┗ 📂imagewoof2-160
   ┣ 📂train
   ┃ ┣ 📂n02086240
   ┃ ┗ ...
   ┗ 📂val
     ┣ 📂n02086240
     ┗ ...
```

### 2. Create Venv

```bash
conda create -n quant python=3.12
conda activate quant
```

### 3. Install PyTorch

Install PyTorch by following the instructions on the [official website](https://pytorch.org/get-started/locally/), replacing `pip3` with `pip`.

Example command:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 4. Install requirements

```bash
pip install -r requirements.txt
```

## Run

### 1. Transfer Learning to Get the Baseline Model

```bash
python -m quant_experiment.transfer_learning
```

### 2. Quantization


```bash
python -m quant_experiment.quant_torch_eager
```

```bash
python -m quant_experiment.quant_dorefa
```

> quanto seems to have bugs, [link](https://github.com/huggingface/optimum-quanto/issues/378)

```bash
python -m quant_experiment.quant_optimum_quanto
```

# 量化可以调的超参数

Pytorch官方只支持均匀量化

可以分别设置对于权重(Weight)和激活(Activation)的量化参数，包括：

1. 使用哪种Observer（用于统计Tensor的数据分布的算法，用于确定均匀量化的min-max范围），包括MinMaxObserver、HistogramObserver、MovingAverageMinMaxObserver和它们对应的PerChannel版本
2. 量化的粒度，包括per-tensor和per-channel（没有全局量化）
3. 量化的数据类型，包括float16、int8、int4等
4. 选择使用对称量化或非对称量化
