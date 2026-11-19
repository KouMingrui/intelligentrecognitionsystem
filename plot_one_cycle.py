import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "condition+monitoring+of+hydraulic+systems"

#格式：周期ID(报错的部件, 真实标签, 预测标签)
error_cases = {
    225: ("valve", 80, 90),
    314: ("accumulator", 130, 115),
    629: ("pump", 2, 1),
    1465: ("cooler", 100, 20),
    1546: ("valve", 100, 90)
}

#针对不同部件选择最能解释错误的通道
def get_channels_for_component(comp):
    if comp == "pump":
        return ["FS1", "EPS1", "PS1"] # 泵坏了看流量和功率
    elif comp == "valve":
        return ["PS1", "PS2", "FS1"] # 阀坏了看压力变化
    elif comp == "cooler":
        return ["TS1", "TS4", "PS1"] # 冷却器看进出水温差
    elif comp == "accumulator":
        return ["PS5", "PS6", "PS1"] # 蓄能器看储能压力
    return ["PS1", "FS1", "TS1"]

CHANNELS_FS = {"PS1":100, "PS2":100, "PS3":100, "PS4":100, "PS5":100, "PS6":100, 
               "EPS1":100, "FS1":10, "FS2":10, "TS1":1, "TS2":1, "TS3":1, "TS4":1, "VS1":1}

for cid, (comp, true_val, pred_val) in error_cases.items():
    channels_to_plot = get_channels_for_component(comp)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Error Case: Cycle {cid} | {comp.capitalize()} | True: {true_val} -> Pred: {pred_val}", 
                 fontsize=14, fontweight='bold', color='darkred')
    
    for i, ch in enumerate(channels_to_plot):
        fs = CHANNELS_FS[ch]
        raw = np.loadtxt(f"{DATA_DIR}/{ch}.txt")[cid]
        t = np.arange(len(raw)) / fs
        
        axes[i].plot(t, raw, color=['tab:blue', 'tab:orange', 'tab:green'][i], linewidth=1.5)
        axes[i].set_ylabel(f"{ch} (sampling rate {fs}Hz)")
        axes[i].grid(alpha=0.3)
        
        axes[i].set_facecolor('#f8f9fa')
    
    axes[-1].set_xlabel("Observation Time (s)", fontsize=12)
    plt.tight_layout()
    
    path = f"figures/analysis_cycle_{cid}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight") 
    plt.close()
    print(f"已生成: {path}")