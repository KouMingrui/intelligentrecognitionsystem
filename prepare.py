import numpy as np
import pandas as pd

PROFILE = "condition+monitoring+of+hydraulic+systems/profile.txt"   # 改成你的实际路径
OUT = "manifest.csv"

profile = np.loadtxt(PROFILE)
labels = profile[:, :4].astype(int)
stable = profile[:, 4].astype(int)

# 1. 构建 144 种四元组
groups = sorted(set(tuple(row) for row in labels))
healthy = (100, 100, 0, 130)
groups.remove(healthy)

# 2. 固定种子打乱（任务书原文）
rng = np.random.default_rng(2026)
perm = rng.permutation(len(groups))
groups = [groups[i] for i in perm]

train_groups = groups[:86]
val_groups   = groups[86:115]
test_groups  = groups[115:] + [healthy]

split_map = {}
for g in train_groups: split_map[g] = "train"
for g in val_groups:   split_map[g] = "val"
for g in test_groups:  split_map[g] = "test"

# 3. 生成 manifest
rows = []
for i, (lab, st) in enumerate(zip(labels, stable)):
    g = tuple(lab)
    rows.append({
        "cycle_id": i,
        "cooler": g[0],
        "valve": g[1],
        "pump": g[2],
        "accumulator": g[3],
        "group_id": "_".join(map(str, g)),
        "stable_flag": st,
        "split": split_map[g],
        "original_row": i + 1,
    })
manifest = pd.DataFrame(rows)
manifest.to_csv(OUT, index=False)

# 4. 统计
print("=== 每个集合的组合数与周期数 ===")
print(manifest.groupby("split").agg(
    n_groups=("group_id", "nunique"),
    n_cycles=("cycle_id", "count"),
))

print("\n=== 四部件等级在三个集合中的覆盖 ===")
for col in ["cooler", "valve", "pump", "accumulator"]:
    print(f"\n{col}:")
    print(pd.crosstab(manifest[col], manifest["split"]))

print("\n=== 稳定性标志分布 ===")
print(pd.crosstab(manifest["stable_flag"], manifest["split"]))

print(f"\n总周期数: {len(manifest)}")