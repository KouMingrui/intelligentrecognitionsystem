import numpy as np
import pandas as pd
import json, os
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_DIR = "condition+monitoring+of+hydraulic+systems"
TARGETS = ["cooler", "valve", "pump", "accumulator"]
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}
NORMAL = {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130}

profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
labels = profile[:, :4].astype(int)
stable_flags = profile[:, 4].astype(int)
manifest = pd.read_csv("manifest.csv")
test_idx = manifest.index[manifest["split"] == "test"].values

def evaluate_model(model_type):
    preds_df = pd.read_csv(f"predictions_{model_type}.csv")
    alarms = json.load(open(f"alarms_{model_type}.json"))
    
    res = {}
    #四部件指标与稳定性分层
    for sec in [10, 30, 60]:
        sec_preds = preds_df[preds_df["budget_sec"] == sec]
        macro_f1s = []
        
        for comp in TARGETS:
            comp_preds = sec_preds[sec_preds["component"] == comp].set_index("cycle_id")
            y_true = labels[test_idx, LABEL_COLS[comp]]
            y_pred = [comp_preds.loc[cid, "pred_level"] for cid in test_idx]
            
            #计算该部件的 Macro-F1
            f1 = f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true), zero_division=0)
            macro_f1s.append(f1)
            
        #严格联合准确率
        joint_correct = 0
        for cid in test_idx:
            cycle_preds = sec_preds[sec_preds["cycle_id"] == cid]
            is_all_correct = True
            for comp in TARGETS:
                pred = cycle_preds[cycle_preds["component"] == comp]["pred_level"].values[0]
                if pred != labels[cid, LABEL_COLS[comp]]:
                    is_all_correct = False
                    break
            if is_all_correct: joint_correct += 1

        res[f"budget_{sec}s"] = {
            "mean_macro_f1": round(np.mean(macro_f1s), 4),
            "strict_joint_accuracy": round(joint_correct / len(test_idx), 4),
        }

        #稳定性分层
        if sec == 60:
            for flag in [0, 1]:
                flag_idx = test_idx[stable_flags[test_idx] == flag]
                flag_joint_correct = 0
                for cid in flag_idx:
                    cycle_preds = sec_preds[sec_preds["cycle_id"] == cid]
                    if all(cycle_preds[cycle_preds["component"] == comp]["pred_level"].values[0] == labels[cid, LABEL_COLS[comp]] for comp in TARGETS):
                        flag_joint_correct += 1
                res[f"budget_{sec}s"][f"stable_flag_{flag}_joint_acc"] = round(flag_joint_correct / len(flag_idx), 4) if len(flag_idx) else 0

    #异常定位F1
    sec_60_preds = preds_df[preds_df["budget_sec"] == 60]
    y_true_bin, y_pred_bin = [], []
    for comp in TARGETS:
        comp_preds = sec_60_preds[sec_60_preds["component"] == comp].set_index("cycle_id")
        for cid in test_idx:
            tv = labels[cid, LABEL_COLS[comp]]
            pv = comp_preds.loc[cid, "pred_level"]
            y_true_bin.append(0 if tv == NORMAL[comp] else 1)
            y_pred_bin.append(0 if pv == NORMAL[comp] else 1)
    res["anomaly_localization_f1"] = round(f1_score(y_true_bin, y_pred_bin, zero_division=0), 4)

    #告警指标
    alarm_true, alarm_pred, time_30, time_60, missed = [], [], 0, 0, 0
    for cid in test_idx:
        for comp in TARGETS:
            tv = labels[cid, LABEL_COLS[comp]]
            is_truly_abnormal = (tv != NORMAL[comp])
            comp_alarms = [a for a in alarms if a["cycle_id"] == cid and a["component"] == comp]
            has_alarm = len(comp_alarms) > 0
            alarm_true.append(1 if is_truly_abnormal else 0)
            alarm_pred.append(1 if has_alarm else 0)
            if has_alarm and is_truly_abnormal:
                if comp_alarms[0]["alarm_time_sec"] <= 30: time_30 += 1
                else: time_60 += 1
            elif not has_alarm and is_truly_abnormal: missed += 1
    
    res["alarm_metrics"] = {
        "f1": round(f1_score(alarm_true, alarm_pred, zero_division=0), 4),
        "time_30s": time_30, "time_60s": time_60, "missed": missed
    }
    return res

if __name__ == "__main__":
    final_metrics = {}
    for mt in ["b0", "b1", "b2", "b3", "b4"]:
        if os.path.exists(f"predictions_{mt}.csv"):
            final_metrics[mt] = evaluate_model(mt)
    
    with open("metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
    print("已生成完整的 metrics.json，包含 B0/B1/B2/B3/B4 评测与稳定性分层！")