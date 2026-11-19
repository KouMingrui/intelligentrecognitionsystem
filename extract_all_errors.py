import pandas as pd
import numpy as np
import os

DATA_DIR = "condition+monitoring+of+hydraulic+systems"
TARGETS = ["cooler", "valve", "pump", "accumulator"]
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}
NORMAL = {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130}

profile = np.loadtxt(f"{DATA_DIR}/profile.txt")
labels = profile[:, :4].astype(int)
manifest = pd.read_csv("manifest.csv")
test_idx = manifest.index[manifest["split"] == "test"].values

#用来存三个模型的错误 DataFrame
error_dfs = {}

for model_type in ["b0", "b1", "b3"]:
    try:
        preds = pd.read_csv(f"predictions_{model_type}.csv")
    except FileNotFoundError:
        print(f"跳过 {model_type}，未找到预测文件。")
        continue
        
    preds_60 = preds[preds["budget_sec"] == 60]
    errors = []
    
    for cid in test_idx:
        for comp in TARGETS:
            pred = preds_60[(preds_60["cycle_id"] == cid) & (preds_60["component"] == comp)]["pred_level"].values[0]
            true = labels[cid, LABEL_COLS[comp]]
            
            if pred != true:
                #判断是误报还是漏报
                is_truly_normal = (true == NORMAL[comp])
                err_type = "误报(False Alarm)" if is_truly_normal else "漏报/判错(Misclassification)"
                
                errors.append({
                    "cycle_id": cid,
                    "component": comp,
                    "true_label": true,
                    "predicted_label": pred,
                    "error_type": err_type
                })
                
    df_err = pd.DataFrame(errors)
    df_err.to_csv(f"errors_{model_type}.csv", index=False)
    print(f"[{model_type.upper()}] 共找到 {len(errors)} 个错误，已保存到 errors_{model_type}.csv")
    error_dfs[model_type] = df_err

#寻找B0、B1、B3同时出错的困难样本
if "b0" in error_dfs and "b1" in error_dfs and "b3" in error_dfs:
    print("\n开始寻找三个模型同时出错的困难样本...")
    
    #以cycle_id和component为键进行内连接
    common = pd.merge(error_dfs["b0"], error_dfs["b1"], on=["cycle_id", "component"], suffixes=("_b0", "_b1"))
    common = pd.merge(common, error_dfs["b3"], on=["cycle_id", "component"])
    
    
    common = common.rename(columns={
        "true_label": "true_label_b3",
        "predicted_label": "predicted_label_b3",
        "error_type": "error_type_b3"
    })
    
    common.to_csv("errors_common.csv", index=False)
    print(f"发现 {len(common)} 个三个模型均出错的困难样本，已保存到 errors_common.csv！")
    
    if len(common) > 0:
        print("\n=== 部分困难样本展示 ===")
        print(common.head())