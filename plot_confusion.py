import numpy as np
import pandas as pd
import json, os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#配置
DATA_DIR = "condition+monitoring+of+hydraulic+systems"
BUDGETS = [10, 30, 60]
TARGETS = ["cooler", "valve", "pump", "accumulator"]
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}
CHANNELS = {
    "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100,
    "EPS1": 100,
    "FS1": 10, "FS2": 10,
    "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1,
}
FIG_DIR = "figures"

#特征提取
def extract_features(x):
    n = x.shape[1]
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
    return np.column_stack([mean, std, mn, mx, rms, slope]), near_const.astype(int)

#检查模型目录
for seconds in BUDGETS:
    d = f"models/b0_{seconds}s"
    if not os.path.isdir(d):
        raise FileNotFoundError(f"找不到 {d}，请先运行 train_b0.py 生成三个预算的模型")

#读原始数据
print("读入原始数据...")
raw = {ch: np.loadtxt(f"{DATA_DIR}/{ch}.txt") for ch in CHANNELS}
profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
labels = profile[:, :4].astype(int)
manifest = pd.read_csv("manifest.csv")
idx = {s: manifest.index[manifest["split"] == s].values for s in ["val", "test"]}
print("读入完成")

os.makedirs(FIG_DIR, exist_ok=True)

#预计算特征
X_by_budget = {}
for seconds in BUDGETS:
    out_dir = f"models/b0_{seconds}s"
    meta = json.load(open(f"{out_dir}/meta.json"))
    med = np.array(meta["impute_median"])

    feat_list, miss_list = [], []
    for ch, fs in CHANNELS.items():
        x = raw[ch][:, :fs * seconds]
        f, miss = extract_features(x)
        feat_list.append(f)
        miss_list.append(miss.reshape(-1, 1))
    X = np.hstack(feat_list + miss_list)

    X_imp = X.copy()
    inds = np.where(np.isnan(X_imp))
    X_imp[inds] = med[inds[1]]
    X_by_budget[seconds] = X_imp

#画图
for tgt in TARGETS:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    fig.subplots_adjust(left=0.08, right=0.85, bottom=0.08, top=0.88, wspace=0.35, hspace=0.35)

    last_im = None
    for col, seconds in enumerate(BUDGETS):
        out_dir = f"models/b0_{seconds}s"
        meta = json.load(open(f"{out_dir}/meta.json"))
        clf = joblib.load(f"{out_dir}/rf_{tgt}.joblib")
        classes = meta["label_mapping"][tgt]
        X = X_by_budget[seconds]

        for row, split in enumerate(["val", "test"]):
            y_true = labels[idx[split], LABEL_COLS[tgt]]
            y_pred = clf.predict(X[idx[split]])
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

            ax = axes[row, col]
            last_im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

            for i in range(len(classes)):
                for j in range(len(classes)):
                    val = cm[i, j]
                    ratio = cm_norm[i, j]
                    txt = f"{val}\n{ratio:.2f}"
                    color = "white" if ratio > 0.5 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=9, color=color)

            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, fontsize=10)
            ax.set_yticklabels(classes, fontsize=10)
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("True", fontsize=11)
            ax.set_title(f"{seconds}s | {split}", fontsize=12, pad=10)

    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) 
    fig.colorbar(last_im, cax=cbar_ax, label="Row-normalized (recall)")

    fig.suptitle(f"B0 Confusion Matrix — {tgt}", fontsize=16, y=0.95)

    path = f"{FIG_DIR}/cm_{tgt}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存 {path}")

print(f"\n全部完成，四张图在 {FIG_DIR}/ 目录下")