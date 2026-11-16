import numpy as np
import pandas as pd
import json, os
import joblib
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

#配置
DATA_DIR = "condition+monitoring+of+hydraulic+systems"
BUDGETS = [10, 30, 60]
TARGETS = ["cooler", "valve", "pump", "accumulator"]
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}
NORMAL = {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130}

CHANNELS = {
    "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100,
    "EPS1": 100,
    "FS1": 10, "FS2": 10,
    "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1,
}

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

#读数据
print("读入原始数据...")
raw = {ch: np.loadtxt(f"{DATA_DIR}/{ch}.txt") for ch in CHANNELS}
manifest = pd.read_csv("manifest.csv")
test_idx = manifest.index[manifest["split"] == "test"].values

#加载模型与批量预测
print("加载模型并执行批量预测...")
models = {}
batch_preds = {}   # 存储批量预测的等级
batch_probas = {}  # 存储批量预测的概率

for seconds in BUDGETS:
    out_dir = f"models/b0_{seconds}s"
    meta = json.load(open(f"{out_dir}/meta.json"))
    med = np.array(meta["impute_median"])
    
    models[seconds] = {}
    batch_preds[seconds] = {}
    batch_probas[seconds] = {}

    #计算特征
    feat_list, miss_list = [], []
    for ch, fs in CHANNELS.items():
        x = raw[ch][:, :fs * seconds]
        f, miss = extract_features(x)
        feat_list.append(f)
        miss_list.append(miss.reshape(-1, 1))
    X = np.hstack(feat_list + miss_list)
    
    #补偿
    X_imp = X.copy()
    inds = np.where(np.isnan(X_imp))
    X_imp[inds] = med[inds[1]]
    
    # 提取测试集全体特征
    X_test = X_imp[test_idx]

    for comp in TARGETS:
        clf = joblib.load(f"{out_dir}/rf_{comp}.joblib")
        models[seconds][comp] = clf
        #一次性预测全部测试集避免在循环里反复调用
        batch_preds[seconds][comp] = clf.predict(X_test)
        batch_probas[seconds][comp] = clf.predict_proba(X_test)

#告警逻辑模拟
print("开始时间线回放与告警判定...")
predictions = []
alarms = []

for i, cycle_id in enumerate(tqdm(test_idx, desc="处理进度")):
    candidates = {}
    alarmed_components = set()
    cycle_updates = {tgt: [] for tgt in TARGETS}

    for time in BUDGETS:
        for comp in TARGETS:
            #直接从刚才算好的数组里查表，不再调用模型
            pred = batch_preds[time][comp][i]
            proba = batch_probas[time][comp][i]
            classes = list(models[time][comp].classes_)
            score = proba[classes.index(pred)] if pred in classes else 0.0
            
            predictions.append({
                "cycle_id": int(cycle_id),
                "budget_sec": time,
                "component": comp,
                "pred_level": int(pred),
                "score": round(float(score), 4)
            })

            if time == 10:
                cycle_updates[comp] = [{"time": 10, "level": int(pred)}]
            else:
                cycle_updates[comp].append({"time": time, "level": int(pred)})

            is_abnormal = (pred != NORMAL[comp])
            
            if is_abnormal:
                if comp not in candidates:
                    candidates[comp] = (pred, time)
                elif candidates[comp][0] == pred:
                    if comp not in alarmed_components:
                        alarms.append({
                            "cycle_id": int(cycle_id),
                            "component": comp,
                            "alarm_time_sec": time,
                            "alarm_level": int(pred),
                            "candidate_time_sec": candidates[comp][1],
                            "final_level": int(pred),
                            "updates": cycle_updates[comp]
                        })
                        alarmed_components.add(comp)
                else:
                    candidates[comp] = (pred, time)
            else:
                if comp in candidates:
                    del candidates[comp]
    
    #更新final_level
    for alarm in alarms:
        if alarm["cycle_id"] == cycle_id:
            alarm["final_level"] = cycle_updates[alarm["component"]][-1]["level"]

#保存结果
pd.DataFrame(predictions).to_csv("predictions.csv", index=False)
print("已保存 predictions.csv")

with open("alarms.json", "w", encoding="utf-8") as f:
    json.dump(alarms, f, indent=2, ensure_ascii=False)
print(f"已保存 alarms.json，共 {len(alarms)} 条正式告警。")