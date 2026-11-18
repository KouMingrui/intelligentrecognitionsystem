import numpy as np
import pandas as pd
import json, os
import joblib
import argparse

DATA_DIR = "condition+monitoring+of+hydraulic+systems"
BUDGETS = [10, 30, 60]
TARGETS = ["cooler", "valve", "pump", "accumulator"]
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}
NORMAL = {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130}
CHANNELS = {"PS1":100,"PS2":100,"PS3":100,"PS4":100,"PS5":100,"PS6":100,"EPS1":100,"FS1":10,"FS2":10,"TS1":1,"TS2":1,"TS3":1,"TS4":1,"VS1":1}

#复用三种特征提取
def extract_b0(x):
    n = x.shape[1]
    mean, std = x.mean(1), x.std(1)
    mn, mx = x.min(1), x.max(1)
    rms = np.sqrt((x**2).mean(1))
    t = np.arange(n, dtype=float); t -= t.mean()
    slope = (x*t).sum(1)/(t**2).sum()
    near_const = std < 1e-8; slope[near_const] = np.nan
    return np.column_stack([mean,std,mn,mx,rms,slope]), near_const.astype(int)

def extract_b1(x):
    n = x.shape[1]
    f_b0, miss = extract_b0(x)
    if n >= 15:
        s_len = n//3; s1, s2, s3 = x[:,:s_len], x[:,s_len:2*s_len], x[:,2*s_len:3*s_len]
        m1, m2, m3 = s1.mean(1), s2.mean(1), s3.mean(1)
        d12, d23 = m1-m2, m2-m3
    else:
        m1=m2=m3=d12=d23=np.zeros(x.shape[0])
    return np.column_stack([f_b0,m1,m2,m3,d12,d23]), miss

def extract_b2(x):
    n = x.shape[1]
    f_b0, miss = extract_b0(x)
    if n >= 15:
        s_len = n//3; s1, s2, s3 = x[:,:s_len], x[:,s_len:2*s_len], x[:,2*s_len:3*s_len]
        m1, m2, m3 = s1.mean(1), s2.mean(1), s3.mean(1)
    else:
        m1=m2=m3=np.zeros(x.shape[0])
    return np.column_stack([f_b0,m1,m2,m3]), miss

EXTRACT_FNS = {"b0": extract_b0, "b1": extract_b1, "b2": extract_b2}

def run_alarm_eval(model_type):
    print(f"评估模型: {model_type.upper()}")
    extract_fn = EXTRACT_FNS[model_type]
    raw = {ch: np.loadtxt(f"{DATA_DIR}/{ch}.txt") for ch in CHANNELS}
    manifest = pd.read_csv("manifest.csv")
    test_idx = manifest.index[manifest["split"] == "test"].values

    models, batch_preds, batch_probas = {}, {}, {}
    for seconds in BUDGETS:
        out_dir = f"models/{model_type}_{seconds}s"
        meta = json.load(open(f"{out_dir}/meta.json"))
        med = np.array(meta["impute_median"])
        models[seconds], batch_preds[seconds], batch_probas[seconds] = {}, {}, {}
        
        feat_list, miss_list = [], []
        for ch, fs in CHANNELS.items():
            x = raw[ch][:, :fs*seconds]
            f, m = extract_fn(x)
            feat_list.append(f)
            miss_list.append(m.reshape(-1, 1))
        X = np.hstack(feat_list + miss_list)
        X_imp = X.copy()
        inds = np.where(np.isnan(X_imp))
        X_imp[inds] = med[inds[1]]
        X_test = X_imp[test_idx]

        for comp in TARGETS:
            clf = joblib.load(f"{out_dir}/rf_{comp}.joblib")
            models[seconds][comp] = clf
            batch_preds[seconds][comp] = clf.predict(X_test)
            batch_probas[seconds][comp] = clf.predict_proba(X_test)

    predictions, alarms = [], []
    for i, cycle_id in enumerate(test_idx):
        candidates, alarmed_components, cycle_updates = {}, set(), {t: [] for t in TARGETS}
        for time in BUDGETS:
            for comp in TARGETS:
                pred = batch_preds[time][comp][i]
                proba = batch_probas[time][comp][i]
                classes = list(models[time][comp].classes_)
                score = proba[classes.index(pred)] if pred in classes else 0.0
                
                predictions.append({
                    "cycle_id": int(cycle_id), "budget_sec": time, "component": comp,
                    "pred_level": int(pred), "score": round(float(score), 4)
                })
                
                cycle_updates[comp].append({"time": time, "level": int(pred)})
                is_abnormal = (pred != NORMAL[comp])
                
                if is_abnormal:
                    if comp not in candidates: candidates[comp] = (pred, time)
                    elif candidates[comp][0] == pred:
                        if comp not in alarmed_components:
                            alarms.append({
                                "cycle_id": int(cycle_id), "component": comp,
                                "alarm_time_sec": time, "alarm_level": int(pred),
                                "candidate_time_sec": candidates[comp][1], "final_level": int(pred),
                                "updates": cycle_updates[comp]
                            })
                            alarmed_components.add(comp)
                    else: candidates[comp] = (pred, time)
                else:
                    if comp in candidates: del candidates[comp]
        for alarm in alarms:
            if alarm["cycle_id"] == cycle_id:
                alarm["final_level"] = cycle_updates[alarm["component"]][-1]["level"]

    pd.DataFrame(predictions).to_csv(f"predictions_{model_type}.csv", index=False)
    with open(f"alarms_{model_type}.json", "w", encoding="utf-8") as f:
        json.dump(alarms, f, indent=2, ensure_ascii=False)
    print(f"已保存 predictions_{model_type}.csv 和 alarms_{model_type}.json (共 {len(alarms)} 条告警)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="b0", choices=["b0", "b1", "b2"])
    args = parser.parse_args()
    run_alarm_eval(args.model)