import numpy as np
import pandas as pd
import json, os
import joblib
import argparse
import time
import matplotlib.pyplot as plt

# 配置
DATA_DIR = "condition+monitoring+of+hydraulic+systems"
BUDGETS = [10, 30, 60]
TARGETS = ["cooler", "valve", "pump", "accumulator"]
NORMAL = {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130}
CHANNELS = {"PS1":100,"PS2":100,"PS3":100,"PS4":100,"PS5":100,"PS6":100,"EPS1":100,"FS1":10,"FS2":10,"TS1":1,"TS2":1,"TS3":1,"TS4":1,"VS1":1}
REP_CHANNELS = ["PS1", "FS1", "TS1", "VS1"] # 代表通道：压力、流量、温度、振动

def extract_b1(x):
    n = x.shape[1]
    mean, std, mn, mx = x.mean(1), x.std(1), x.min(1), x.max(1)
    rms = np.sqrt((x**2).mean(1))
    t = np.arange(n, dtype=float); t -= t.mean()
    slope = (x*t).sum(1)/(t**2).sum()
    near_const = std < 1e-8; slope[near_const] = np.nan
    if n >= 15:
        s_len = n//3; s1, s2, s3 = x[:,:s_len], x[:,s_len:2*s_len], x[:,2*s_len:3*s_len]
        m1, m2, m3 = s1.mean(1), s2.mean(1), s3.mean(1)
        d12, d23 = m1-m2, m2-m3
    else:
        m1=m2=m3=d12=d23=np.zeros(x.shape[0])
    return np.column_stack([mean,std,mn,mx,rms,slope,m1,m2,m3,d12,d23]), near_const.astype(int)

def extract_b0(x):
    n = x.shape[1]
    mean, std, mn, mx = x.mean(1), x.std(1), x.min(1), x.max(1)
    rms = np.sqrt((x**2).mean(1))
    t = np.arange(n, dtype=float); t -= t.mean()
    slope = (x*t).sum(1)/(t**2).sum()
    near_const = std < 1e-8; slope[near_const] = np.nan
    return np.column_stack([mean,std,mn,mx,rms,slope]), near_const.astype(int)

def extract_b3(x):
    return extract_b1(x)

def extract_b4(x):
    return extract_b0(x)

def run_replay(cycle_id, model_type, speed, mask_fs1):
    print(f"\n▶ 开始回放 周期 {cycle_id} | 模型: {model_type.upper()} | 速度: {speed} | 屏蔽FS1: {mask_fs1}")
    
    extract_fn_map = {"b0": extract_b0, "b1": extract_b1, "b2": extract_b1, "b3": extract_b3, "b4": extract_b4}
    extract_fn = extract_fn_map.get(model_type, extract_b1)
    
    # 准备绘图数据
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.canvas.manager.set_window_title(f'Cycle {cycle_id} Replay')
    plot_data = {}
    for i, ch in enumerate(REP_CHANNELS):
        ax = axes[i//2, i%2]
        raw = np.loadtxt(f"{DATA_DIR}/{ch}.txt")[cycle_id]
        plot_data[ch] = raw
        ax.plot(raw, color='lightgray', label='Full Cycle (Ground Truth)')
        ax.set_title(f"{ch} (fs={CHANNELS[ch]}Hz)")
        ax.grid(True, alpha=0.3)
        
        # 如果屏蔽了FS1，在图上标红提示
        if mask_fs1 and ch == "FS1":
            ax.text(0.5, 0.5, "SENSOR OFFLINE", transform=ax.transAxes, 
                    ha='center', va='center', fontsize=15, color='red', fontweight='bold')
            
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    
    # 加载模型
    models, metas = {}, {}
    for sec in BUDGETS:
        out_dir = f"models/{model_type}_{sec}s"
        metas[sec] = json.load(open(f"{out_dir}/meta.json"))
        models[sec] = {tgt: joblib.load(f"{out_dir}/rf_{tgt}.joblib") for tgt in TARGETS}
    
    candidates, alarmed_components = {}, set()
    
    # 模拟时间流逝
    for sec in BUDGETS:
        time.sleep(2 if speed == "normal" else 0.5) 
        print(f"\n --- 源时间到达: {sec} 秒 ---")
        
        # 在图上画出已观测部分（蓝线）
        for i, ch in enumerate(REP_CHANNELS):
            ax = axes[i//2, i%2]
            if len(ax.lines) > 1: 
                ax.lines[-1].remove()
            
            n_obs = CHANNELS[ch] * sec
            # 如果传感器被屏蔽，蓝线不画（或者画成NaN）
            if mask_fs1 and ch == "FS1":
                pass # 蓝线不画
            else:
                ax.plot(plot_data[ch][:n_obs], color='blue', label='Observed')
            ax.legend()
            
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.5)
        
        # 提取特征
        med = np.array(metas[sec]["impute_median"])
        feat_list, miss_list = [], []
        missing_sensors = []
        
        # 用来存当前截断的数据，供计算物理特征用
        cycle_data = {} 
        
        for ch, fs in CHANNELS.items():
            raw = np.loadtxt(f"{DATA_DIR}/{ch}.txt")[cycle_id:cycle_id+1, :fs*sec]
            
            # 处理传感器屏蔽逻辑
            if mask_fs1 and ch == "FS1":
                raw = np.full_like(raw, np.nan) # 强制变成全NaN
                
            cycle_data[ch] = raw
            
            f, m = extract_fn(raw)
            feat_list.append(f)
            miss_list.append(m.reshape(-1, 1))
            
            # 检查是否是NaN或近常数
            if m[0] == 1 or np.isnan(raw).all(): 
                missing_sensors.append(ch)
                
        X = np.hstack(feat_list + miss_list)
        
        # B3/B4 需要添加物理交叉特征
        if model_type in ["b3", "b4"]:
            eps1_mean = cycle_data["EPS1"].mean(axis=1)
            fs1_mean = cycle_data["FS1"].mean(axis=1) 
            ts1_mean = cycle_data["TS1"].mean(axis=1)
            ts4_mean = cycle_data["TS4"].mean(axis=1)
            
            energy_per_flow = eps1_mean / (fs1_mean + 1e-5)
            temp_diff = ts1_mean - ts4_mean
            X = np.hstack([X, energy_per_flow.reshape(-1,1), temp_diff.reshape(-1,1)])
        
        # 补偿缺失值
        X = np.where(np.isnan(X), med, X)
        
        if missing_sensors:
            print(f"警告：检测到缺失/掉线传感器 -> {missing_sensors}")
            
        # 预测与告警逻辑
        for comp in TARGETS:
            clf = models[sec][comp]
            pred = clf.predict(X)[0]
            proba = clf.predict_proba(X)[0]
            score = proba[list(clf.classes_).index(pred)]
            
            status = "正常" if pred == NORMAL[comp] else "异常"
            msg = ""
            if pred != NORMAL[comp]:
                if comp not in candidates:
                    candidates[comp] = pred; msg = "候选"
                elif candidates[comp] == pred:
                    if comp not in alarmed_components:
                        msg = "触发正式告警！"; alarmed_components.add(comp)
                    else: msg = "(已告警，更新)"
                else:
                    candidates[comp] = pred; msg = "候选更新"
            else:
                if comp in candidates: del candidates[comp]
            
            print(f"  [{comp:12s}] 预测: {pred:3d} ({status}) | 得分: {score:.2f} {msg}")
            
    print("\n回放结束。关闭窗口退出。")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, default=100)
    parser.add_argument("--model", type=str, default="b1", choices=["b0", "b1", "b2", "b3", "b4"])
    parser.add_argument("--speed", type=str, default="normal", choices=["normal", "fast"])
    parser.add_argument("--mask_fs1", action="store_true", help="模拟FS1传感器掉线")
    args = parser.parse_args()
    
    run_replay(args.cycle, args.model, args.speed, args.mask_fs1)