import numpy as np
import pandas as pd
import json, os
import joblib

#配置
DATA_DIR = "condition+monitoring+of+hydraulic+systems"
BUDGETS = [10, 30, 60]
TARGETS = ["cooler", "valve", "pump", "accumulator"]
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}

#各部件的正常等级定义
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
profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
manifest = pd.read_csv("manifest.csv")

#只在测试集上生成告警和预测结果
test_idx = manifest.index[manifest["split"] == "test"].values

#加载模型与特征
print("加载模型...")
models = {}
X_by_budget = {}

for seconds in BUDGETS:
    out_dir = f"models/b0_{seconds}s"
    meta = json.load(open(f"{out_dir}/meta.json"))
    med = np.array(meta["impute_median"])
    
    models[seconds] = {}
    for tgt in TARGETS:
        models[seconds][tgt] = joblib.load(f"{out_dir}/rf_{tgt}.joblib")

    #计算特征
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

#告警逻辑模拟
print("开始时间线回放与告警判定...")
predictions = []  # 存储predictions.csv
alarms = []       # 存储alarms.json

for cycle_id in test_idx:
    #每个周期重置状态
    candidates = {}  # 记录当前候选{component: (level, time)}
    alarmed_components = set() # 已告警过的部件，保证最多一次
    cycle_updates = {tgt: [] for tgt in TARGETS} # 记录后续更新记录

    #时间线推进
    for time in BUDGETS:
        X_cycle = X_by_budget[time][cycle_id:cycle_id+1]
        
        for comp in TARGETS:
            clf = models[time][comp]
            pred = clf.predict(X_cycle)[0]
            proba = clf.predict_proba(X_cycle)[0]
            
            #找出预测等级对应的概率得分
            classes = list(clf.classes_)
            score = proba[classes.index(pred)] if pred in classes else 0.0
            
            #记录预测结果
            predictions.append({
                "cycle_id": int(cycle_id),
                "budget_sec": time,
                "component": comp,
                "pred_level": int(pred),
                "score": round(float(score), 4)
            })

            # 记录该时点的更新轨迹（用于 alarms.json）
            if time == 10:
                cycle_updates[comp] = [{"time": 10, "level": int(pred)}]
            else:
                cycle_updates[comp].append({"time": time, "level": int(pred)})

            #判定逻辑
            is_abnormal = (pred != NORMAL[comp])
            
            if is_abnormal:
                if comp not in candidates:
                    #10秒第一次发现异常，记录候选
                    candidates[comp] = (pred, time)
                elif candidates[comp][0] == pred:
                    #连续两个时点预测出了相同的异常等级
                    if comp not in alarmed_components:
                        #触发正式告警
                        alarms.append({
                            "cycle_id": int(cycle_id),
                            "component": comp,
                            "alarm_time_sec": time,
                            "alarm_level": int(pred),
                            "candidate_time_sec": candidates[comp][1],
                            "final_level": int(pred), #稍后覆盖为60秒的最终预测
                            "updates": cycle_updates[comp]
                        })
                        alarmed_components.add(comp)
                        #不重复计数已经告警过了如果后面还是异常不需要再次告警
                else:
                    # 异常等级发生了变化更新候选
                    candidates[comp] = (pred, time)
            else:
                # 恢复正常清除该部件的候选状态
                if comp in candidates:
                    del candidates[comp]
    
    # 周期结束更新这一周期所有告警记录的final_level
    for alarm in alarms:
        if alarm["cycle_id"] == cycle_id:
            # 该部件在60秒时的最终预测
            alarm["final_level"] = cycle_updates[alarm["component"]][-1]["level"]

#保存结果
#保存 predictions.csv
pd.DataFrame(predictions).to_csv("predictions.csv", index=False)
print("已保存 predictions.csv")

#保存 alarms.json
with open("alarms.json", "w", encoding="utf-8") as f:
    json.dump(alarms, f, indent=2, ensure_ascii=False)
print(f"已保存 alarms.json，共 {len(alarms)} 条正式告警。")