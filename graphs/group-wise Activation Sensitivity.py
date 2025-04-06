import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取 CSV 数据
df = pd.read_csv("/Users/grexrr/Documents/quant_experiment/runs/resnet18/decompose_quant/tboard/results.csv")

# 清洗数据
df = df.dropna(subset=["quant_weight", "quant_act", "test_acc"])
df["decompose_method"] = df["decompose_method"].fillna("baseline")

# 去掉 CP 和 CP-finetune 方法
df = df[~df["decompose_method"].str.contains("cp")]

# 绘图
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df,
    x="quant_act",
    y="test_acc",
    hue="quant_weight",     # 分组：不同的 weight 精度
    errorbar=None           # 不显示误差线（error bar）
)

plt.title("Test Accuracy by Activation Quantization (Non-CP Models)")
plt.xlabel("Activation Quantization Type")
plt.ylabel("Test Accuracy")
plt.legend(title="Weight Bitwidth")
plt.tight_layout()
plt.show()
