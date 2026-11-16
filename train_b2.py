import numpy as np
import pandas as pd
import json, os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

#配置
DATA_DIR = "condition+monitoring+of+hydraulic+systems"
SEED = 2026
BUDGETS = [10, 30, 60]
CHANNELS = {
    "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100,
    "EPS1": 100, "FS1": 10, "FS2": 10,
    "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1,
}
TARGETS = ["cooler", "valve", "pump", "accumulator"]
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}

#消融差分特征
def extract_features_b2(x):
    #B2只保留全段统计和3个分段均值（去掉了2个差分特征）
    n = x.shape[1]
    
    #基础全段统计
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    rms = np.sqrt((x ** 2).mean(axis=1))
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    slope = (x * t).sum(axis=1) / (t ** 2).sum()
    near_const = std < 1e-8
    slope[near_const] = np.nan
    
    #分段均值保留但不做差分
    if n >= 15: 
        seg_len = n // 3
        s1, s2, s3 = x[:, :seg_len], x[:, seg_len:2*seg_len], x[:, 2*seg_len:3*seg_len]
        seg1_mean = s1.mean(axis=1)
        seg2_mean = s2.mean(axis=1)
        seg3_mean = s3.mean(axis=1)
    else:
        seg1_mean = seg2_mean = seg3_mean = np.zeros(x.shape[0])
        
    # 拼接9个特征（6个全局和3个分段）
    features = np.column_stack([
        mean, std, mn, mx, rms, slope,
        seg1_mean, seg2_mean, seg3_mean
    ])
    return features, near_const.astype(int)

#训练流程
print("读入原始数据...")
raw = {ch: np.loadtxt(f"{DATA_DIR}/{ch}.txt") for ch in CHANNELS}
profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
labels = profile[:, :4].astype(int)
manifest = pd.read_csv("manifest.csv")
idx = {s: manifest.index[manifest["split"] == s].values for s in ["train", "val", "test"]}

summary = []

for seconds in BUDGETS:
    print(f"\n{'='*55}\n[方案 B2 - 消融差分] 观测预算: {seconds} 秒\n{'='*55}")
    out_dir = f"models/b2_{seconds}s"
    os.makedirs(out_dir, exist_ok=True)

    feat_list, miss_list, feat_names = [], [], []
    for ch, fs in CHANNELS.items():
        n_use = fs * seconds
        x = raw[ch][:, :n_use]
        f, miss = extract_features_b2(x)
        feat_list.append(f)
        miss_list.append(miss.reshape(-1, 1))
        # 特征名变成 9 个
        feat_names += [f"{ch}_{s}" for s in ["mean", "std", "min", "max", "rms", "slope", "s1_mean", "s2_mean", "s3_mean"]]
    
    for ch in CHANNELS:
        feat_names += [f"{ch}_missing"]

    X = np.hstack(feat_list + miss_list)
    print("特征矩阵:", X.shape)

    med = np.nanmedian(X[idx["train"]], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_imp = X.copy()
    inds = np.where(np.isnan(X_imp))
    X_imp[inds] = med[inds[1]]

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
            print(f"[{tgt}] {split}  Macro-F1={f1:.4f}")
            
            summary.append({
                "seconds": seconds, "target": tgt, "split": split,
                "macro_f1": round(float(f1), 4)
            })

    meta = {
        "seconds": seconds, "seed": SEED, "channels": CHANNELS,
        "feature_names": feat_names, "label_mapping": label_mapping,
        "impute_median": med.tolist(),
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

pd.DataFrame(summary).to_csv("b2_summary.csv", index=False)
print("\n已保存 b2_summary.csv 和模型文件。")