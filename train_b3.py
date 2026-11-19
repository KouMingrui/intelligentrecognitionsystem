import numpy as np
import pandas as pd
import json, os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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

def extract_features_b1(x):
    #B1的特征提取：全段统计分段统计差分
    n = x.shape[1]
    mean, std = x.mean(1), x.std(1)
    mn, mx = x.min(1), x.max(1)
    rms = np.sqrt((x**2).mean(1))
    t = np.arange(n, dtype=float); t -= t.mean()
    slope = (x*t).sum(1)/(t**2).sum()
    near_const = std < 1e-8; slope[near_const] = np.nan
    if n >= 15:
        s_len = n//3; s1, s2, s3 = x[:,:s_len], x[:,s_len:2*s_len], x[:,2*s_len:3*s_len]
        m1, m2, m3 = s1.mean(1), s2.mean(1), s3.mean(1)
        d12, d23 = m1-m2, m2-m3
    else:
        m1=m2=m3=d12=d23=np.zeros(x.shape[0])
    return np.column_stack([mean,std,mn,mx,rms,slope,m1,m2,m3,d12,d23]), near_const.astype(int)

print("读入原始数据...")
raw = {ch: np.loadtxt(f"{DATA_DIR}/{ch}.txt") for ch in CHANNELS}
profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
labels = profile[:, :4].astype(int)
manifest = pd.read_csv("manifest.csv")
idx = {s: manifest.index[manifest["split"] == s].values for s in ["train", "val", "test"]}

summary = []

for seconds in BUDGETS:
    print(f"\n{'='*55}\n[方案 B3 - B1特征 + 物理交叉] 观测预算: {seconds} 秒\n{'='*55}")
    out_dir = f"models/b3_{seconds}s"
    os.makedirs(out_dir, exist_ok=True)

    feat_list, miss_list, feat_names = [], [], []
    for ch, fs in CHANNELS.items():
        x = raw[ch][:, :fs*seconds]
        f, m = extract_features_b1(x)
        feat_list.append(f)
        miss_list.append(m.reshape(-1, 1))
        feat_names += [f"{ch}_{s}" for s in ["mean", "std", "min", "max", "rms", "slope", "s1_mean", "s2_mean", "s3_mean", "diff12", "diff23"]]
    
    for ch in CHANNELS:
        feat_names += [f"{ch}_missing"]

    X = np.hstack(feat_list + miss_list)
    
    #计算物理交叉特征
    eps1_mean = raw["EPS1"][:, :100*seconds].mean(axis=1)
    fs1_mean = raw["FS1"][:, :10*seconds].mean(axis=1)
    ts1_mean = raw["TS1"][:, :1*seconds].mean(axis=1)
    ts4_mean = raw["TS4"][:, :1*seconds].mean(axis=1)
    
    energy_per_flow = eps1_mean / (fs1_mean + 1e-5)
    temp_diff = ts1_mean - ts4_mean
    
    X = np.hstack([X, energy_per_flow.reshape(-1,1), temp_diff.reshape(-1,1)])
    feat_names += ["energy_per_flow", "temp_diff"]
    
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

pd.DataFrame(summary).to_csv("b3_summary.csv", index=False)
print("\n已保存 b3_summary.csv 和模型文件。")