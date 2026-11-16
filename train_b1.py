import numpy as np
import pandas as pd
import json, os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix

#配置
DATA_DIR = "condition+monitoring+of+hydraulic+systems"
SEED = 2026
BUDGETS = [10, 30, 60]

CHANNELS = {
    "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100,
    "EPS1": 100,
    "FS1": 10, "FS2": 10,
    "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1,
}

TARGETS = ["cooler", "valve", "pump", "accumulator"]
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}

#分段特征提取
def extract_features_b1(x):
    """x: (n_cycles, n_samples)"""
    n = x.shape[1]
    
    #基础全段统计
    mean = x.mean(axis=1) #均值
    std = x.std(axis=1) #标准差
    mn = x.min(axis=1) #最小值
    mx = x.max(axis=1) #最大值
    rms = np.sqrt((x ** 2).mean(axis=1)) #线性趋势
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    slope = (x * t).sum(axis=1) / (t ** 2).sum()
    near_const = std < 1e-8 #近常数标记
    slope[near_const] = np.nan
    
    # 分段统计与差分
    # 如果窗口太短就不分段直接补0
    if n >= 15: 
        seg_len = n // 3 #分三段
        seg1 = x[:, :seg_len]
        seg2 = x[:, seg_len:2*seg_len]
        seg3 = x[:, 2*seg_len:3*seg_len]
        
        seg1_mean = seg1.mean(axis=1)
        seg2_mean = seg2.mean(axis=1)
        seg3_mean = seg3.mean(axis=1)

        #算均值的差分
        diff_1_2 = seg1_mean - seg2_mean 
        diff_2_3 = seg2_mean - seg3_mean
    else:
        seg1_mean = seg2_mean = seg3_mean = np.zeros(x.shape[0])
        diff_1_2 = diff_2_3 = np.zeros(x.shape[0])
        
    # 拼接11个特征
    features = np.column_stack([
        mean, std, mn, mx, rms, slope,
        seg1_mean, seg2_mean, seg3_mean, diff_1_2, diff_2_3
    ])
    return features, near_const.astype(int)

# 开始训练流程
print("读入原始数据...")
raw = {ch: np.loadtxt(f"{DATA_DIR}/{ch}.txt") for ch in CHANNELS}
profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
labels = profile[:, :4].astype(int)
manifest = pd.read_csv("manifest.csv")
idx = {s: manifest.index[manifest["split"] == s].values for s in ["train", "val", "test"]}

summary = []

for seconds in BUDGETS:
    print(f"\n{'='*55}\n[方案 B1] 观测预算: {seconds} 秒\n{'='*55}")
    out_dir = f"models/b1_{seconds}s"
    os.makedirs(out_dir, exist_ok=True)

    feat_list, miss_list, feat_names = [], [], []
    for ch, fs in CHANNELS.items():
        n_use = fs * seconds
        x = raw[ch][:, :n_use]
        f, miss = extract_features_b1(x)
        feat_list.append(f)
        miss_list.append(miss.reshape(-1, 1))
        # 特征名有11个
        feat_names += [f"{ch}_{s}" for s in ["mean", "std", "min", "max", "rms", "slope", "s1_mean", "s2_mean", "s3_mean", "diff12", "diff23"]]
    
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
            rec = recall_score(y_true, y_pred, average=None, labels=np.unique(y))
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

pd.DataFrame(summary).to_csv("b1_summary.csv", index=False)
print("\n已保存 b1_summary.csv 和模型文件。")