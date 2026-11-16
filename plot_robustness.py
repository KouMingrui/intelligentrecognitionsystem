import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
df = pd.read_csv("robustness_summary.csv")

conditions = ["clean", "noise_0.05", "noise_0.10", "mask_FS1", "mask_FS1_FS2"]
targets = ["cooler", "valve", "pump", "accumulator"]
models = ["B0", "B1", "B2"]
colors = {"B0": "gray", "B1": "blue", "B2": "orange"}
markers = {"B0": "o", "B1": "s", "B2": "^"}

#60秒主结果五种测试条件对比
df_60 = df[df["seconds"] == 60]

for tgt in targets:
    plt.figure(figsize=(10, 6))
    for model in models:
        sub = df_60[(df_60["model"] == model) & (df_60["target"] == tgt)]
        # 按照 conditions 顺序排列
        sub = sub.set_index("condition").reindex(conditions)
        plt.plot(conditions, sub["macro_f1"], marker=markers[model], 
                 label=model, color=colors[model], linewidth=2)
    
    plt.title(f"{tgt.upper()} Macro-F1 under 5 Conditions (60s Budget)")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=15)
    
    path = f"figures/robustness_{tgt}_60s.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存 {path}")

print("全部图表绘制完成！")