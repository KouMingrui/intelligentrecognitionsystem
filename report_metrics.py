import numpy as np
import pandas as pd
import json, os
from sklearn.metrics import precision_score, recall_score, f1_score

# 配置常量
DATA_DIR = "condition+monitoring+of+hydraulic+systems"
TARGETS = ["cooler", "valve", "pump", "accumulator"]
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}
NORMAL = {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130}
HEALTHY = (100, 100, 0, 130)  # 全正常组合（唯一）
MODEL_TYPES = ["b0", "b1", "b2", "b3", "b4"]
MAIN_BUDGET = 60  # 主结果观测预算（秒）

# 读取基础数据
profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
labels = profile[:, :4].astype(int)
manifest = pd.read_csv("manifest.csv")
test_idx = manifest.index[manifest["split"] == "test"].values
healthy_group_id = "_".join(map(str, HEALTHY))
healthy_idx = manifest.index[
    (manifest["split"] == "test") & (manifest["group_id"] == healthy_group_id)
].values
n_healthy = len(healthy_idx)
print(f"测试集周期数: {len(test_idx)} | 全正常组合周期数: {n_healthy}")


def df_to_md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def calc_false_alarm():
    rows = []
    for mt in MODEL_TYPES:
        preds_path = f"predictions_{mt}.csv"
        alarm_path = f"alarms_{mt}.json"
        if not os.path.exists(preds_path):
            print(f"  跳过 {mt.upper()}，未找到 {preds_path}")
            continue

        preds_60 = pd.read_csv(preds_path)
        preds_60 = preds_60[preds_60["budget_sec"] == MAIN_BUDGET]

        # 部件粒度误报：全正常周期里被预测为非正常值的 (周期, 部件) 数量
        fa_comp = 0
        # 周期粒度误报：四个部件中任意一个误判的周期数
        fa_cycle = 0
        for cid in healthy_idx:
            cyc = preds_60[preds_60["cycle_id"] == cid]
            cycle_has_fa = False
            for comp in TARGETS:
                pv = int(cyc[cyc["component"] == comp]["pred_level"].values[0])
                if pv != NORMAL[comp]:
                    fa_comp += 1
                    cycle_has_fa = True
            if cycle_has_fa:
                fa_cycle += 1

        # 正式告警粒度误报：全正常周期中触发了正式告警的周期数
        alarm_fa_cycle = 0
        if os.path.exists(alarm_path):
            alarms = json.load(open(alarm_path))
            alerted_cycles = set(a["cycle_id"] for a in alarms)
            alarm_fa_cycle = len(set(healthy_idx.tolist()) & alerted_cycles)

        rows.append({
            "模型": mt.upper(),
            "全正常周期数": n_healthy,
            "误报周期数(周期粒度)": fa_cycle,
            "误报率(周期粒度)": f"{fa_cycle}/{n_healthy} = {fa_cycle / n_healthy * 100:.1f}%",
            "误报部件数(部件粒度)": fa_comp,
            "误报率(部件粒度)": f"{fa_comp}/{n_healthy * len(TARGETS)} = {fa_comp / (n_healthy * len(TARGETS)) * 100:.1f}%",
            "正式误报告警周期数": alarm_fa_cycle,
        })
    return pd.DataFrame(rows)


def calc_alarm_dist():
    metrics = json.load(open("metrics.json"))
    rows = []
    for mt in MODEL_TYPES:
        if mt not in metrics:
            continue
        am = metrics[mt].get("alarm_metrics", {})
        t30, t60 = am.get("time_30s", 0), am.get("time_60s", 0)
        total = t30 + t60
        rows.append({
            "模型": mt.upper(),
            "Alarm_F1": am.get("f1"),
            "30秒内告警数": t30,
            "60秒告警数": t60,
            "正确告警总数": total,
            "30秒内占比": f"{t30 / total * 100:.1f}%" if total else "0%",
            "漏报数": am.get("missed"),
        })
    return pd.DataFrame(rows)


def calc_recall_table():
    rows = []
    for mt in MODEL_TYPES:
        preds_path = f"predictions_{mt}.csv"
        if not os.path.exists(preds_path):
            continue
        preds_60 = pd.read_csv(preds_path)
        preds_60 = preds_60[preds_60["budget_sec"] == MAIN_BUDGET]
        for comp in TARGETS:
            comp_preds = preds_60[preds_60["component"] == comp].set_index("cycle_id")
            y_true = labels[test_idx, LABEL_COLS[comp]]
            y_pred = np.array([int(comp_preds.loc[cid, "pred_level"]) for cid in test_idx])
            for lvl in sorted(set(int(v) for v in y_true)):
                mask = y_true == lvl
                rec = float((y_pred[mask] == lvl).mean())
                rows.append({
                    "model": mt.upper(), "component": comp, "level": lvl,
                    "n_samples": int(mask.sum()), "recall": round(rec, 4),
                })
    return pd.DataFrame(rows)


def calc_memory():
    df = pd.read_csv("benchmark_results.csv").copy()
    df["total_peak_mem_mb"] = (df["feat_peak_mem_mb"] + df["inf_peak_mem_mb"]).round(2)
    df = df.rename(columns={
        "model": "模型", "seconds": "秒预算",
        "feat_peak_mem_mb": "特征提取内存峰值(MB)",
        "inf_peak_mem_mb": "模型推断内存峰值(MB)",
        "total_peak_mem_mb": "合计内存峰值(MB)",
    })
    return df[["模型", "秒预算", "特征提取内存峰值(MB)",
               "模型推断内存峰值(MB)", "合计内存峰值(MB)"]]


def calc_anomaly_localization():
    """§8.2：逐部件把状态转正常/异常，计算 Precision/Recall/F1（60秒）"""
    rows = []
    for mt in MODEL_TYPES:
        preds_path = f"predictions_{mt}.csv"
        if not os.path.exists(preds_path):
            continue
        preds_60 = pd.read_csv(preds_path)
        preds_60 = preds_60[preds_60["budget_sec"] == MAIN_BUDGET]
        for comp in TARGETS:
            comp_preds = preds_60[preds_60["component"] == comp].set_index("cycle_id")
            y_true = labels[test_idx, LABEL_COLS[comp]]
            y_pred = np.array([int(comp_preds.loc[cid, "pred_level"]) for cid in test_idx])
            yt = (y_true != NORMAL[comp]).astype(int)
            yp = (y_pred != NORMAL[comp]).astype(int)
            rows.append({
                "模型": mt.upper(), "部件": comp,
                "Precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
                "Recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
                "F1": round(float(f1_score(yt, yp, zero_division=0)), 4),
            })
    return pd.DataFrame(rows)


def calc_alarm_metrics():
    """§8.3：告警 Precision/Recall/F1、健康部件误报率、正式告警等级准确率"""
    rows = []
    for mt in MODEL_TYPES:
        alarm_path = f"alarms_{mt}.json"
        if not os.path.exists(alarm_path):
            continue
        alarms = json.load(open(alarm_path))
        alarm_pairs = set((a["cycle_id"], a["component"]) for a in alarms)

        y_true, y_pred = [], []
        for cid in test_idx:
            for comp in TARGETS:
                tv = labels[cid, LABEL_COLS[comp]]
                y_true.append(1 if tv != NORMAL[comp] else 0)
                y_pred.append(1 if (cid, comp) in alarm_pairs else 0)

        p = float(precision_score(y_true, y_pred, zero_division=0))
        r = float(recall_score(y_true, y_pred, zero_division=0))
        f = float(f1_score(y_true, y_pred, zero_division=0))

        n_normal = sum(1 for v in y_true if v == 0)
        n_fa = sum(1 for t, q in zip(y_true, y_pred) if t == 0 and q == 1)
        fa_rate = n_fa / n_normal if n_normal else 0.0

        lvl_ok = sum(1 for a in alarms
                     if a["alarm_level"] == labels[a["cycle_id"], LABEL_COLS[a["component"]]])
        lvl_acc = lvl_ok / len(alarms) if alarms else 0.0

        rows.append({
            "模型": mt.upper(),
            "告警Precision": round(p, 4),
            "告警Recall": round(r, 4),
            "告警F1": round(f, 4),
            "健康部件误报率": f"{n_fa}/{n_normal} = {fa_rate * 100:.1f}%",
            "告警等级准确率": round(lvl_acc, 4),
        })
    return pd.DataFrame(rows)


def main():
    print("\n>>> 1. 统计全正常组合误报数...")
    fa_df = calc_false_alarm()

    print(">>> 2. 汇总告警时长分布...")
    alarm_df = calc_alarm_dist()

    print(">>> 3. 复算各等级 Recall 与样本数...")
    recall_df = calc_recall_table()
    recall_df.to_csv("recall_table.csv", index=False)

    print(">>> 4. 汇总内存峰值...")
    mem_df = calc_memory()

    print(">>> 5. 计算逐部件异常定位 P/R/F1（8.2）...")
    loc_df = calc_anomaly_localization()
    loc_df.to_csv("anomaly_localization.csv", index=False)

    print(">>> 6. 计算完整告警指标 P/R/F1 与健康部件误报率（8.3）...")
    alarm_metric_df = calc_alarm_metrics()

    # Markdown 报告
    lines = ["# 报告缺失指标汇总\n",
             "> 数据来源：metrics.json / predictions_*.csv / alarms_*.json / manifest.csv / benchmark_results.csv",
             f"> 测试集周期数：{len(test_idx)}　全正常组合周期数：{n_healthy}\n",
             "## 1. 全正常组合误报数（8.2）\n", df_to_md(fa_df),
             "\n## 2. 告警时长分布（8.3）\n", df_to_md(alarm_df),
             "\n## 3. 各方案内存峰值（8.5）\n", df_to_md(mem_df),
             "\n## 4. 各等级 Recall（8.1）\n",
             "> 明细见 `recall_table.csv`（含每等级测试样本数 n_samples）。\n",
             df_to_md(recall_df[recall_df["model"] == "B1"]),
             "\n（以上仅展示 B1，其余方案见 recall_table.csv）\n",
             "\n## 5. 逐部件异常定位 P/R/F1（8.2，60秒）\n", df_to_md(loc_df),
             "\n## 6. 告警完整指标（8.3，60秒）\n", df_to_md(alarm_metric_df), "\n"]

    report = "\n".join(str(x) for x in lines)
    with open("report_metrics.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print("\n已保存 report_metrics.md、recall_table.csv、anomaly_localization.csv")


if __name__ == "__main__":
    main()
