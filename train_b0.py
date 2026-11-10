import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix

DATA_DIR = "condition+monitoring+of+hydraulic+systems"
SECONDS = 60

CHANNELS = {
    "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100,
    "EPS1": 100,
    "FS1": 10, "FS2": 10,
    "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1,
}

def extract_features(x):
    """x: (n_cycles, n_samples) -> (n_cycles, 6)"""
    n = x.shape[1]
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    rms = np.sqrt((x ** 2).mean(axis=1))
    t = np.arange(n)
    t = t - t.mean()
    slope = (x * t).sum(axis=1) / (t ** 2).sum()
    return np.column_stack([mean, std, mn, mx, rms, slope])

# 1. 读标签和划分
profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
labels = profile[:, :4].astype(int)
manifest = pd.read_csv("manifest.csv")

# 2. 逐通道提取特征
feat_list = []
feat_names = []
for ch, fs in CHANNELS.items():
    x = np.loadtxt(f"{DATA_DIR}/{ch}.txt")
    n_use = fs * SECONDS
    x = x[:, :n_use]
    f = extract_features(x)
    feat_list.append(f)
    feat_names += [f"{ch}_{s}" for s in ["mean","std","min","max","rms","slope"]]

X = np.hstack(feat_list)
print("特征矩阵:", X.shape)   # 应为 (2205, 84)

# 3. 按 manifest 的 split 取索引
idx = {s: manifest.index[manifest["split"] == s].values for s in ["train","val","test"]}

# 4. 四个独立随机森林
targets = ["cooler", "valve", "pump", "accumulator"]
label_cols = {"cooler":0, "valve":1, "pump":2, "accumulator":3}

for tgt in targets:
    y = labels[:, label_cols[tgt]]
    clf = RandomForestClassifier(n_estimators=200, random_state=2026, n_jobs=-1)
    clf.fit(X[idx["train"]], y[idx["train"]])

    for split in ["val", "test"]:
        y_pred = clf.predict(X[idx[split]])
        y_true = y[idx[split]]
        f1 = f1_score(y_true, y_pred, average="macro", labels=np.unique(y))
        rec = recall_score(y_true, y_pred, average=None, labels=np.unique(y))
        print(f"\n[{tgt}] {split}  Macro-F1={f1:.4f}")
        print("  各等级 Recall:", dict(zip(np.unique(y), np.round(rec, 4))))
        print("  混淆矩阵:\n", confusion_matrix(y_true, y_pred, labels=np.unique(y)))