low_rank_decompose
==================

> Downloaded and modified from the repo [alibaba/MNN](https://github.com/alibaba/MNN)

# 1. 整体流程

`low_rank_decompose`函数通过对PyTorch的`Conv2d`和`Linear`层进行低秩分解，实现模型压缩。其中，`Linear`层采用SVD分解，`Conv2d`层采用Tucker分解(SVD分解的高维拓展)。

# 2. Conv2d层的Tucker分解

## 2.1 适用条件

- 普通卷积层`Conv2d`（`groups=1`且非1x1卷积核）。
- 输入/输出通道数需大于`align_channels`，避免无效分解。

## 2.2 分解步骤

1. 权重展开与秩估计：
   - 使用`VBMF.EVBMF`（基于变分贝叶斯矩阵分解）估计输入通道维度（`mode=1`）和输出通道维度（`mode=0`）的秩。
   - 最终秩由`get_align_channels`确定，确保对齐并满足最小压缩比例`tucker_minimal_ratio`。
2. Tucker分解：
   - 调用`partial_tucker`将权重张量分解为核心张量（`core`）和两个因子矩阵（`first`和`last`）。
   - 分解公式：  
     \[
     W \approx \text{core} \times_1 \text{last} \times_2 \text{first}
     \]
   - 原始卷积层被替换为三个新层：
     - 1x1卷积（调整输入通道）：`first_layer`（权重为`first`的转置）。
     - 核心卷积：`core_layer`（权重为`core`，保留原始卷积核大小）。
     - 1x1卷积（调整输出通道）：`last_layer`（权重为`last`，保留原始偏置）。
3. 参数对齐：
   - 分解后的通道数需为`align_channels`的倍数，例如输入通道从64对齐到64或72（若`align_channels=8`）。

# 3. Linear层与1x1 Conv2d的SVD分解

## 3.1 适用条件

- `Linear`层或等效的1x1 `Conv2d`（无空间维度操作）。

## 3.2 分解步骤

1. 奇异值截断：
   - 对权重矩阵进行SVD分解：\( W = U \Sigma V^T \)。
   - 根据`reserved_singular_value_ratio`确定保留的奇异值数量`n_dim`，使得前`n_dim`个奇异值之和占总和的指定比例。
   - `n_dim`需对齐成`align_channels`的倍数，优化硬件计算效率。
2. 近似重构：
   - 分解为两个子层：
     - 降维层：\( W_1 = \Sigma_{1:n\_\text{dim}} V_{1:n\_\text{dim}}^T \)，减少输入维度至`n_dim`。
     - 升维层：\( W_2 = U_{1:n\_\text{dim}} \)，恢复原始输出维度。
   - 对`Conv2d`，重构为两个1x1卷积；对`Linear`，重构为两个全连接层。

# 4. 关键技术细节

- 秩的选择：
  - `VBMF`自动估计秩，避免手动调参。
  - `tucker_minimal_ratio`确保分解后的通道不低于原始通道的指定比例。
  - `reserved_singular_value_ratio`控制保留的奇异值能量。
- 硬件友好设计：
  - `align_channels`确保分解后的张量维度对齐，适配GPU/NPU的并行计算。

# 5. 压缩效果

- 参数量计算：
  - 原始`Conv2d`参数量：\( C_\text{out} \times C_\text{in} \times K_h \times K_w \)。
  - 分解后参数量：\( C_\text{out} \times R + R \times C_\text{in} + R \times R \times K_h \times K_w \)（Tucker分解，\( R \)为秩）。
  - 显著减少大卷积核的参数量（如3x3卷积）。

# 6. 应用场景

- 模型轻量化：适用于移动端或边缘设备部署。
- 加速推理：分解后的轻量层可减少计算量。
- 兼容性：与量化、剪枝等其他压缩方法协同使用。
