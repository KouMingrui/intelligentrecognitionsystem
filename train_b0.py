import numpy as np
import pandas as pd
import json, os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix

DATA_DIR = "condition+monitoring+of+hydraulic+systems"
OUT_DIR = "models/b0"
#可更改的观测预算
SECONDS = 60
SEED = 2026

CHANNELS = {
    "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100,
    "EPS1": 100,
    "FS1": 10, "FS2": 10,
    "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1,
}

def extract_features(x):
    """返回 (n_cycles, 6) 特征 和 (n_cycles,) 近常数标记"""
    n = x.shape[1]
    mean = x.mean(axis=1) #均值
    std = x.std(axis=1) #标准差
    mn = x.min(axis=1) #最小值
    mx = x.max(axis=1) #最大值
    rms = np.sqrt((x ** 2).mean(axis=1)) #均方根
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    slope = (x * t).sum(axis=1) / (t ** 2).sum() #线性趋势
    near_const = std < 1e-8          # 近常数通道使用标准差判定周期内几乎不变
    slope[near_const] = np.nan        # slope无意义标记为缺失
    return np.column_stack([mean, std, mn, mx, rms, slope]), near_const.astype(int)

# 读标签和划分
profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
labels = profile[:, :4].astype(int)
manifest = pd.read_csv("manifest.csv")
idx = {s: manifest.index[manifest["split"] == s].values for s in ["train","val","test"]}

# 逐通道提取特征和缺失标记
feat_list, miss_list, feat_names = [], [], []
for ch, fs in CHANNELS.items():
    x = np.loadtxt(f"{DATA_DIR}/{ch}.txt")[:, :fs*SECONDS]
    f, miss = extract_features(x)
    feat_list.append(f)
    miss_list.append(miss.reshape(-1, 1))
    feat_names += [f"{ch}_{s}" for s in ["mean","std","min","max","rms","slope"]]
    feat_names += [f"{ch}_missing"]

X = np.hstack(feat_list + miss_list)
print("特征矩阵:", X.shape)

# 训练集中位数补偿
med = np.nanmedian(X[idx["train"]], axis=0)
med = np.where(np.isnan(med), 0.0, med)

def impute(X, med):
    X2 = X.copy()
    inds = np.where(np.isnan(X2))
    X2[inds] = med[inds[1]]
    return X2

X_imp = impute(X, med)
print("补偿后剩余 NaN:", np.isnan(X_imp).sum())

# 训练并保存
os.makedirs(OUT_DIR, exist_ok=True)
targets = ["cooler", "valve", "pump", "accumulator"]
label_cols = {"cooler":0, "valve":1, "pump":2, "accumulator":3}
label_mapping = {}

for tgt in targets:
    y = labels[:, label_cols[tgt]]
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    clf.fit(X_imp[idx["train"]], y[idx["train"]])
    joblib.dump(clf, f"{OUT_DIR}/rf_{tgt}.joblib")
    label_mapping[tgt] = [int(v) for v in np.unique(y)]

    for split in ["val", "test"]:
        y_pred = clf.predict(X_imp[idx[split]])
        y_true = y[idx[split]]
        f1 = f1_score(y_true, y_pred, average="macro", labels=np.unique(y))
        rec = recall_score(y_true, y_pred, average=None, labels=np.unique(y))
        print(f"\n[{tgt}] {split}  Macro-F1={f1:.4f}")
        print("  Recall:", dict(zip(np.unique(y), np.round(rec, 4))))
        print("  混淆矩阵:\n", confusion_matrix(y_true, y_pred, labels=np.unique(y)))

# 保存元数据 特征顺序补偿参数随机种子标签映射近常数规则
meta = {
    "seconds": SECONDS,
    "seed": SEED,
    "channels": CHANNELS,
    "feature_names": feat_names,
    "label_mapping": label_mapping,
    "impute_median": med.tolist(),
    "near_const_rule": "std < 1e-8 -> slope = NaN",
}
with open(f"{OUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"\n已保存: {OUT_DIR}/ 下 4 个 .joblib + meta.json")