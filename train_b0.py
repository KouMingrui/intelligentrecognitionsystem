import numpy as np
import pandas as pd
import json, os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix

DATA_DIR = "condition+monitoring+of+hydraulic+systems"
SEED = 2026
BUDGETS = [10, 30, 60] #三种时间预算

CHANNELS = {
    "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100,
    "EPS1": 100,
    "FS1": 10, "FS2": 10,
    "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1,
}

TARGETS = ["cooler", "valve", "pump", "accumulator"] #四个输出为冷却器阀泵蓄能器
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}

def extract_features(x):
    """x: (n_cycles, n_samples) -> 6 个统计量 + 近常数标记"""
    n = x.shape[1]
    mean = x.mean(axis=1) #均值
    std = x.std(axis=1) #方差
    mn = x.min(axis=1) #最小值
    mx = x.max(axis=1) #最大值
    rms = np.sqrt((x ** 2).mean(axis=1)) #均方根
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    slope = (x * t).sum(axis=1) / (t ** 2).sum() #线性趋势
    near_const = std < 1e-8 # 近常数使用标准差判断
    slope[near_const] = np.nan
    return np.column_stack([mean, std, mn, mx, rms, slope]), near_const.astype(int)

# 原始数据只读一次
print("读入原始数据...")
raw = {ch: np.loadtxt(f"{DATA_DIR}/{ch}.txt") for ch in CHANNELS}
print("读入完成")

profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
labels = profile[:, :4].astype(int)
manifest = pd.read_csv("manifest.csv")
idx = {s: manifest.index[manifest["split"] == s].values for s in ["train", "val", "test"]}

summary = []

# 三个预算分别独立走完整流程
for seconds in BUDGETS:
    print(f"\n{'='*55}\n观测预算: {seconds} 秒\n{'='*55}")
    out_dir = f"models/b0_{seconds}s"
    os.makedirs(out_dir, exist_ok=True)

    feat_list, miss_list, feat_names = [], [], []
    for ch, fs in CHANNELS.items():
        n_use = fs * seconds
        x = raw[ch][:, :n_use]
        f, miss = extract_features(x)
        feat_list.append(f)
        miss_list.append(miss.reshape(-1, 1))
        #加6个基础特征名
        feat_names += [f"{ch}_{s}" for s in ["mean", "std", "min", "max", "rms", "slope"]]
    
    #循环结束后统一加所有的missing特征名
    for ch in CHANNELS:
        feat_names += [f"{ch}_missing"]

    # 拼接前84列是特征后14列是 missing
    X = np.hstack(feat_list + miss_list)
    print("特征矩阵:", X.shape)

    # 补偿参数只从当前预算的训练集拟合
    med = np.nanmedian(X[idx["train"]], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_imp = X.copy()
    inds = np.where(np.isnan(X_imp))
    X_imp[inds] = med[inds[1]]
    print("补偿后剩余 NaN:", np.isnan(X_imp).sum())

    label_mapping = {}
    for tgt in TARGETS:
        y = labels[:, LABEL_COLS[tgt]]
        clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
        clf.fit(X_imp[idx["train"]], y[idx["train"]])
        joblib.dump(clf, f"{out_dir}/rf_{tgt}.joblib")
        label_mapping[tgt] = [int(v) for v in np.unique(y)]

        for split in ["val", "test"]:
            y_pred = clf.predict(X_imp[idx[split]])
            y_true = y[idx[split]]
            f1 = f1_score(y_true, y_pred, average="macro", labels=np.unique(y))
            rec = recall_score(y_true, y_pred, average=None, labels=np.unique(y))
            print(f"\n[{tgt}] {split}  Macro-F1={f1:.4f}")
            print("  各等级 Recall:", dict(zip(np.unique(y), np.round(rec, 4))))
            print("  混淆矩阵:\n", confusion_matrix(y_true, y_pred, labels=np.unique(y)))

            summary.append({
                "seconds": seconds, "target": tgt, "split": split,
                "macro_f1": round(float(f1), 4),
                **{f"recall_{int(c)}": round(float(r), 4)
                   for c, r in zip(np.unique(y), rec)},
            })

    meta = {
        "seconds": seconds, "seed": SEED, "channels": CHANNELS,
        "feature_names": feat_names, "label_mapping": label_mapping,
        "impute_median": med.tolist(),
        "near_const_rule": "std < 1e-8 -> slope = NaN",
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"已保存: {out_dir}/")

# 三预算汇总表
summary_df = pd.DataFrame(summary)
summary_df.to_csv("b0_summary.csv", index=False)
print("\n=== 三预算 Macro-F1 汇总（行=部件+集合，列=预算）===")
pivot = summary_df.pivot_table(index=["target", "split"], columns="seconds", values="macro_f1")
print(pivot)
print("\n已保存 b0_summary.csv")