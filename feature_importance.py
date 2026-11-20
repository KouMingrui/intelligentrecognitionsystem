import json
import numpy as np
import pandas as pd
import joblib

# 配置（与训练脚本保持一致）
TARGETS = ["cooler", "valve", "pump", "accumulator"]
CHANNELS = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1",
            "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1"]
BUDGETS = [10, 30, 60]
MODEL = "b1"   # 使用改进方案 B1（含分段统计与差分特征）


def channel_of(feat_name):
    """把特征名映射回所属通道，如 'PS1_s2_mean' -> 'PS1'。"""
    for ch in CHANNELS:
        if feat_name.startswith(ch + "_"):
            return ch
    return "unknown"


rows = []
for sec in BUDGETS:
    meta = json.load(open(f"models/{MODEL}_{sec}s/meta.json"))
    names = meta["feature_names"]
    for tgt in TARGETS:
        clf = joblib.load(f"models/{MODEL}_{sec}s/rf_{tgt}.joblib")
        imp = clf.feature_importances_
        agg = {}
        for nm, v in zip(names, imp):
            ch = channel_of(nm)
            agg[ch] = agg.get(ch, 0.0) + float(v)
        assert len(names) == len(imp), "特征名与重要性长度不一致"
        for ch, v in agg.items():
            rows.append({"budget_sec": sec, "component": tgt,
                         "channel": ch, "importance": v})

df = pd.DataFrame(rows)
df.to_csv("feature_importance_by_channel.csv", index=False)

# 汇总：四部件取平均后按通道聚合，对比不同预算
mean_df = df.groupby(["budget_sec", "channel"])["importance"].mean().reset_index()
piv = mean_df.pivot(index="channel", columns="budget_sec", values="importance")
piv = piv.sort_values(60, ascending=False)
piv.columns = [f"{c}s" for c in piv.columns]

print("=== 各通道平均特征重要性（B1，四部件取平均）===")
print(piv.round(4).to_string())
print("\n=== 前 5 重要通道（按 10s 排序）===")
print(piv.sort_values("10s", ascending=False).head(5).round(4).to_string())

# 阀门是性能瓶颈部件，单独查看
valve = df[df.component == "valve"].pivot(
    index="channel", columns="budget_sec", values="importance")
valve = valve.sort_values(60, ascending=False)
valve.columns = [f"{c}s" for c in valve.columns]
print("\n=== 阀门部件的各通道特征重要性 ===")
print(valve.round(4).to_string())

print("\n已保存 feature_importance_by_channel.csv")
