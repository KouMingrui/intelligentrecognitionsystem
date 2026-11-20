import pandas as pd
import numpy as np

# 配置
TARGETS = ["cooler", "valve", "pump", "accumulator"]
NORMAL = {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130}
BUDGET = 60
BASE, IMPROVED = "b0", "b1"   # 基线与改进方案

manifest = pd.read_csv("manifest.csv")
test = manifest[manifest["split"] == "test"].copy()
truth = test.set_index("cycle_id")[TARGETS]
group = test.set_index("cycle_id")["group_id"]


def load_pred(model):
    df = pd.read_csv(f"predictions_{model}.csv")
    df = df[df["budget_sec"] == BUDGET]
    wide = df.pivot(index="cycle_id", columns="component", values="pred_level")
    return wide[TARGETS]


pred_base = load_pred(BASE)
pred_imp = load_pred(IMPROVED)

pred_base = pred_base.loc[truth.index]
pred_imp = pred_imp.loc[truth.index]

res = pd.DataFrame(index=truth.index)
res["group_id"] = group
res["base_joint"] = (pred_base == truth).all(axis=1).astype(int)
res["imp_joint"] = (pred_imp == truth).all(axis=1).astype(int)
for comp in TARGETS:
    res[f"base_{comp}"] = (pred_base[comp] == truth[comp]).astype(int)
    res[f"imp_{comp}"] = (pred_imp[comp] == truth[comp]).astype(int)

# 按状态组合聚合
agg = res.groupby("group_id").agg(
    n=("base_joint", "size"),
    base_joint=("base_joint", "mean"),
    imp_joint=("imp_joint", "mean"),
)
agg["delta_joint"] = agg["imp_joint"] - agg["base_joint"]

# 标注组合类型
def parse(gid):
    return tuple(int(x) for x in gid.split("_"))

agg["valve_anomaly"] = agg.index.to_series().apply(lambda g: parse(g)[1] != NORMAL["valve"])
agg["pump_anomaly"] = agg.index.to_series().apply(lambda g: parse(g)[2] != NORMAL["pump"])
agg["has_anomaly"] = agg.index.to_series().apply(lambda g: parse(g) != (100, 100, 0, 130))
agg = agg.sort_values("delta_joint", ascending=False)
agg.to_csv("improvement_by_group.csv")

print("=== 按状态组合聚合的联合准确率与 B1-B0 增益（60 秒）===")
print(agg.round(3).to_string())

print("\n=== 逐部件整体增益（全部测试周期平均）===")
for comp in TARGETS:
    a = res[f"base_{comp}"].mean()
    b = res[f"imp_{comp}"].mean()
    print(f"{comp:12s}  B0={a:.3f}  B1={b:.3f}  delta={b - a:+.3f}")

print("\n=== 组合增益（按是否含阀门异常分组）===")
print(agg.groupby("valve_anomaly")["delta_joint"].agg(["count", "mean", "max"]).round(3).to_string())

print("\n=== 组合增益（按是否含泵异常分组）===")
print(agg.groupby("pump_anomaly")["delta_joint"].agg(["count", "mean", "max"]).round(3).to_string())

print("\n=== 增益最高的 8 个组合 ===")
print(agg.head(8).round(3).to_string())
print("\n=== 增益最低（含退化）的 8 个组合 ===")
print(agg.tail(8).round(3).to_string())

print("\n已保存 improvement_by_group.csv")
