import numpy as np
import pandas as pd
import json, os
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_DIR = "condition+monitoring+of+hydraulic+systems"
SEED = 2026
BUDGETS = [10, 30, 60]
TARGETS = ["cooler", "valve", "pump", "accumulator"]
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}
NORMAL = {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130}
CHANNELS = {
    "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100,
    "EPS1": 100, "FS1": 10, "FS2": 10,
    "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1,
}

#特征提取函数

def extract_b0(x):
    n = x.shape[1]
    mean, std = x.mean(axis=1), x.std(axis=1)
    mn, mx = x.min(axis=1), x.max(axis=1)
    rms = np.sqrt((x ** 2).mean(axis=1))
    t = np.arange(n, dtype=float); t = t - t.mean()
    slope = (x * t).sum(axis=1) / (t ** 2).sum()
    near_const = std < 1e-8
    slope[near_const] = np.nan
    return np.column_stack([mean, std, mn, mx, rms, slope]), near_const.astype(int)

def extract_b1(x):
    n = x.shape[1]
    f_b0, miss = extract_b0(x)
    if n >= 15:
        seg_len = n // 3
        s1, s2, s3 = x[:, :seg_len], x[:, seg_len:2*seg_len], x[:, 2*seg_len:3*seg_len]
        m1, m2, m3 = s1.mean(axis=1), s2.mean(axis=1), s3.mean(axis=1)
        d12, d23 = m1 - m2, m2 - m3
    else:
        m1 = m2 = m3 = d12 = d23 = np.zeros(x.shape[0])
    return np.column_stack([f_b0, m1, m2, m3, d12, d23]), miss

def extract_b2(x):
    n = x.shape[1]
    f_b0, miss = extract_b0(x)
    if n >= 15:
        seg_len = n // 3
        s1, s2, s3 = x[:, :seg_len], x[:, seg_len:2*seg_len], x[:, 2*seg_len:3*seg_len]
        m1, m2, m3 = s1.mean(axis=1), s2.mean(axis=1), s3.mean(axis=1)
    else:
        m1 = m2 = m3 = np.zeros(x.shape[0])
    return np.column_stack([f_b0, m1, m2, m3]), miss

def extract_b3(x):
    # B3: B1特征 + 物理交叉特征（物理交叉特征在外部计算）
    return extract_b1(x)

def extract_b4(x):
    # B4: B0特征 + 物理交叉特征（物理交叉特征在外部计算）
    return extract_b0(x)

#主评测逻辑
print("读入原始数据...")
raw = {ch: np.loadtxt(f"{DATA_DIR}/{ch}.txt") for ch in CHANNELS}
profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
labels = profile[:, :4].astype(int)
manifest = pd.read_csv("manifest.csv")
train_idx = manifest.index[manifest["split"] == "train"].values
test_idx = manifest.index[manifest["split"] == "test"].values

# 计算各通道训练集的标准差（用于加噪）
train_stds = {ch: raw[ch][train_idx].std() for ch in CHANNELS}

results = []            # 逐部件 Macro-F1（原口径，保持数值不变）
loc_results = []        # 异常定位 F1（异常 vs 正常，池化所有部件）
alarm_results = []      # 告警 P/R/F1 与健康误报率（周期--部件对口径）
conditions = ["clean", "noise_0.05", "noise_0.10", "mask_FS1", "mask_FS1_FS2"]
model_types = [("b0", extract_b0), ("b1", extract_b1), ("b2", extract_b2), ("b3", extract_b3), ("b4", extract_b4)]


def run_alarm_state_machine(batch_preds):
    """在扰动后的预测上运行连续双时点防抖告警状态机，返回 {(周期下标, 部件): 告警等级}。"""
    alarm_level = {}
    for i in range(len(test_idx)):
        candidates, done = {}, set()
        for time in BUDGETS:
            for comp in TARGETS:
                pred = batch_preds[time][comp][i]
                if pred != NORMAL[comp]:
                    if comp not in candidates:
                        candidates[comp] = pred
                    elif candidates[comp] == pred:
                        if comp not in done:
                            alarm_level[(i, comp)] = int(pred)
                            done.add(comp)
                    else:
                        candidates[comp] = pred
                else:
                    candidates.pop(comp, None)
    return alarm_level


np.random.seed(SEED)

for cond in conditions:
    print(f"\n>>> 正在测试条件: {cond}")
    
    #生成受扰动数据
    raw_perturbed = {}
    for ch in CHANNELS:
        x = raw[ch].copy()
        if cond == "clean":
            pass
        elif cond.startswith("noise"):
            factor = float(cond.split("_")[1])
            noise = np.random.normal(0, factor * train_stds[ch], x.shape)
            x = x + noise
        elif cond == "mask_FS1" and ch == "FS1":
            x = np.full_like(x, np.nan)  #整列变成 NaN
        elif cond == "mask_FS1_FS2" and ch in ["FS1", "FS2"]:
            x = np.full_like(x, np.nan)
        raw_perturbed[ch] = x

    #对五种模型进行提取预测评测（先算三预算预测，再统一评测异常定位与告警）
    for model_type, extract_fn in model_types:
        batch_preds = {s: {} for s in BUDGETS}
        for seconds in BUDGETS:
            out_dir = f"models/{model_type}_{seconds}s"
            meta = json.load(open(f"{out_dir}/meta.json"))
            med = np.array(meta["impute_median"])

            #提取特征
            feat_list, miss_list = [], []
            for ch, fs in CHANNELS.items():
                x = raw_perturbed[ch][:, :fs * seconds]
                f, miss = extract_fn(x)
                feat_list.append(f)
                miss_list.append(miss.reshape(-1, 1))
            X = np.hstack(feat_list + miss_list)
            
            # B3/B4 需要添加物理交叉特征
            if model_type in ["b3", "b4"]:
                eps1_mean = raw_perturbed["EPS1"][:, :100*seconds].mean(axis=1)
                fs1_mean = raw_perturbed["FS1"][:, :10*seconds].mean(axis=1)
                ts1_mean = raw_perturbed["TS1"][:, :1*seconds].mean(axis=1)
                ts4_mean = raw_perturbed["TS4"][:, :1*seconds].mean(axis=1)
                energy_per_flow = eps1_mean / (fs1_mean + 1e-5)
                temp_diff = ts1_mean - ts4_mean
                X = np.hstack([X, energy_per_flow.reshape(-1,1), temp_diff.reshape(-1,1)])
            
            #补偿
            X_imp = X.copy()
            inds = np.where(np.isnan(X_imp))
            X_imp[inds] = med[inds[1]]

            #预测
            for tgt in TARGETS:
                clf = joblib.load(f"{out_dir}/rf_{tgt}.joblib")
                y_true = labels[test_idx, LABEL_COLS[tgt]]
                y_pred = clf.predict(X_imp[test_idx])
                batch_preds[seconds][tgt] = y_pred
                f1 = f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))
                
                results.append({
                    "condition": cond, "model": model_type.upper(), 
                    "seconds": seconds, "target": tgt, "macro_f1": round(f1, 4)
                })

            #异常定位：将(周期,部件)的真实/预测等级二值化为"异常 vs 正常"后池化（口径同 evaluate.py §8.2）
            yt_bin, yp_bin = [], []
            for tgt in TARGETS:
                y_true = labels[test_idx, LABEL_COLS[tgt]]
                y_pred = batch_preds[seconds][tgt]
                for tv, pv in zip(y_true, y_pred):
                    yt_bin.append(0 if tv == NORMAL[tgt] else 1)
                    yp_bin.append(0 if pv == NORMAL[tgt] else 1)
            loc_results.append({
                "condition": cond, "model": model_type.upper(), "seconds": seconds,
                "anomaly_precision": round(precision_score(yt_bin, yp_bin, zero_division=0), 4),
                "anomaly_recall": round(recall_score(yt_bin, yp_bin, zero_division=0), 4),
                "anomaly_loc_f1": round(f1_score(yt_bin, yp_bin, zero_division=0), 4),
            })

        #告警：在扰动预测上跑防抖状态机，按"周期--部件对"口径统计（与 tab:alarm_full 一致）
        alarm_level = run_alarm_state_machine(batch_preds)
        yt_a, yp_a = [], []
        for i, cid in enumerate(test_idx):
            for comp in TARGETS:
                tv = labels[cid, LABEL_COLS[comp]]
                yt_a.append(0 if tv == NORMAL[comp] else 1)
                yp_a.append(1 if (i, comp) in alarm_level else 0)
        n_normal = sum(1 for v in yt_a if v == 0)
        n_fa = sum(1 for t, q in zip(yt_a, yp_a) if t == 0 and q == 1)
        lvl_ok = sum(1 for (i, comp), lv in alarm_level.items()
                     if lv == labels[test_idx[i], LABEL_COLS[comp]])
        alarm_results.append({
            "condition": cond, "model": model_type.upper(),
            "alarm_precision": round(precision_score(yt_a, yp_a, zero_division=0), 4),
            "alarm_recall": round(recall_score(yt_a, yp_a, zero_division=0), 4),
            "alarm_f1": round(f1_score(yt_a, yp_a, zero_division=0), 4),
            "healthy_false_alarm_rate": f"{n_fa}/{n_normal} = {n_fa/n_normal*100:.1f}%" if n_normal else "0",
            "alarm_level_acc": round(lvl_ok/len(alarm_level), 4) if alarm_level else 0.0,
        })

#保存结果
df = pd.DataFrame(results)
df.to_csv("robustness_summary.csv", index=False)
print("\n已保存 robustness_summary.csv")

loc_df = pd.DataFrame(loc_results)
loc_df.to_csv("robustness_localization.csv", index=False)
print("已保存 robustness_localization.csv")

alarm_df = pd.DataFrame(alarm_results)
alarm_df.to_csv("robustness_alarm.csv", index=False)
print("已保存 robustness_alarm.csv")

#打印60秒主结果对比
print("\n=== 鲁棒性实验结果 (60秒主结果) ===")
pivot = df[df["seconds"] == 60].pivot_table(index="target", columns=["model", "condition"], values="macro_f1")
print(pivot)

print("\n=== 扰动下异常定位 F1 (60秒) ===")
print(loc_df[loc_df["seconds"] == 60].pivot_table(index="model", columns="condition", values="anomaly_loc_f1"))

print("\n=== 扰动下告警 F1 (跨三预算状态机) ===")
print(alarm_df.pivot_table(index="model", columns="condition", values="alarm_f1"))