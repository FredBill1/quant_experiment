import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("/Users/grexrr/Documents/quant_experiment/graphs/benchmark_results.csv")

# 衍生字段
df["model_size_MB"] = df["model_size"] / (1024 * 1024)
df["inference_time_ms"] = df["cuda_time_per_image_ns"] / 1e6

# ================================
# 图 1: quant_weight vs 推理时间
# ================================
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df[df["quant_weight"].notna()],  # 排除 baseline
    x="quant_weight",
    y="inference_time_ms"
)
plt.title("Inference Time by Quantization Bitwidth (Weight)")
plt.xlabel("Weight Quantization Bitwidth")
plt.ylabel("Inference Time per Image (ms)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================
# 图 2: decomposition method vs 推理时间
# ================================
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df[df["decompose_method"].notna()],  # 排除 baseline if you want
    x="decompose_method",
    y="inference_time_ms",
    hue="quant_weight"  # 可选: 查看不同量化下的 decomposition 效果
)
plt.title("Inference Time by Decomposition Method")
plt.xlabel("Decomposition Method")
plt.ylabel("Inference Time per Image (ms)")
plt.grid(True)
plt.tight_layout()
plt.show()
