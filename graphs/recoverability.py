import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("/Users/grexrr/Documents/quant_experiment/runs/resnet18/decompose_quant/tboard/results.csv")

# 构造 (method, factor) 的组合键
df["key"] = df["decompose_method"] + "_" + df["decompose_factor"].astype(str)

# 分为 finetune / no-finetune 两组
df_ft = df[df["do_finetune"] == True]
df_nft = df[df["do_finetune"] == False]

# 创建 recoverability 数据结构
recover_dict = {}

for key in df["key"].unique():
    acc_ft = df_ft[df_ft["key"] == key]["test_acc"]
    acc_nft = df_nft[df_nft["key"] == key]["test_acc"]

    # 如果两组都有数据，计算平均差值
    if not acc_ft.empty and not acc_nft.empty:
        recover_dict[key] = acc_ft.mean() - acc_nft.mean()

# 拆解 key 为 method 和 factor
methods = []
factors = []
recovers = []

for key, val in recover_dict.items():
    method, factor = key.split("_")
    methods.append(method)
    factors.append(float(factor))
    recovers.append(val)

recover_df = pd.DataFrame({
    "method": methods,
    "factor": factors,
    "recoverability": recovers
})


plt.figure(figsize=(10, 6))
sns.lineplot(
    data=recover_df,
    x="factor",
    y="recoverability",
    hue="method",
    marker="o"
)
plt.title("Recoverability vs. Decomposition Factor")
plt.xlabel("Decomposition Factor")
plt.ylabel("Recoverability (Test Acc Gain After Fine-tune)")
plt.grid(True)
plt.tight_layout()
plt.show()
