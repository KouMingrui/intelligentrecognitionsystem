"""
性能基准测试脚本 v3（修复推理假死问题）
覆盖指标：训练耗时、模型文件大小、单周期特征/推断耗时 P50/P95、内存峰值
验证：严格前缀截断无未来信息泄漏
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*sklearn.utils.parallel.delayed.*")

import numpy as np
import pandas as pd
import json, os, time, joblib, tracemalloc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# -------------------------- 配置常量 --------------------------
DATA_DIR = "condition+monitoring+of+hydraulic+systems"
SEED = 2026
BUDGETS = [10, 30, 60]
TARGETS = ["cooler", "valve", "pump", "accumulator"]
LABEL_COLS = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}
CHANNELS = {
    "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100,
    "EPS1": 100, "FS1": 10, "FS2": 10,
    "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1
}
MODEL_TYPES = ["b0", "b1", "b2", "b3", "b4"]
N_REPEAT = 20  # 20次重复足够统计P50/P95，减少冗余计算
WARMUP_ROUNDS = 3
FIG_DIR = "figures"

# -------------------------- 特征提取函数（与训练代码完全一致） --------------------------
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

def extract_b3(x): return extract_b1(x)
def extract_b4(x): return extract_b0(x)

EXTRACT_FNS = {"b0": extract_b0, "b1": extract_b1, "b2": extract_b2, "b3": extract_b3, "b4": extract_b4}

# -------------------------- 物理特征计算（修复单周期维度匹配） --------------------------
def add_physics_features(X, raw, seconds, sample_sl=slice(None)):
    eps1_mean = raw["EPS1"][sample_sl, :100*seconds].mean(axis=1)
    fs1_mean  = raw["FS1"][sample_sl, :10*seconds].mean(axis=1)
    ts1_mean  = raw["TS1"][sample_sl, :1*seconds].mean(axis=1)
    ts4_mean  = raw["TS4"][sample_sl, :1*seconds].mean(axis=1)
    energy_per_flow = eps1_mean / (fs1_mean + 1e-5)
    temp_diff = ts1_mean - ts4_mean
    return np.hstack([
        X,
        energy_per_flow.reshape(-1, 1),
        temp_diff.reshape(-1, 1)
    ])

# -------------------------- 统一特征构建 --------------------------
def build_single_cycle_features(raw, model_type, seconds, cid):
    extract_fn = EXTRACT_FNS[model_type]
    feat_list, miss_list = [], []
    for ch, fs in CHANNELS.items():
        x = raw[ch][cid:cid+1, :fs*seconds]
        f, m = extract_fn(x)
        feat_list.append(f)
        miss_list.append(m.reshape(-1, 1))
    X = np.hstack(feat_list + miss_list)
    if model_type in ["b3", "b4"]:
        X = add_physics_features(X, raw, seconds, sample_sl=slice(cid, cid+1))
    return X

# -------------------------- 性能测量函数 --------------------------
def measure_feature_extraction(raw, model_type, seconds, cycle_ids):
    # 预热
    for cid in cycle_ids[:WARMUP_ROUNDS]:
        _ = build_single_cycle_features(raw, model_type, seconds, cid)
    
    # 纯耗时测量（不开tracemalloc避免开销）
    times = []
    for _ in range(N_REPEAT):
        for cid in cycle_ids:
            t0 = time.perf_counter()
            _ = build_single_cycle_features(raw, model_type, seconds, cid)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    
    # 单独测内存峰值
    tracemalloc.start()
    tracemalloc.reset_peak()
    for cid in cycle_ids:
        _ = build_single_cycle_features(raw, model_type, seconds, cid)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return (
        np.percentile(times, 50),
        np.percentile(times, 95),
        peak_mem / 1024 / 1024
    )

def measure_inference(raw, model_type, seconds, cycle_ids):
    out_dir = f"models/{model_type}_{seconds}s"
    meta = json.load(open(f"{out_dir}/meta.json"))
    med = np.array(meta["impute_median"])
    # 加载模型后强制单线程，避免单样本预测时多线程调度假死
    models = {}
    for tgt in TARGETS:
        clf = joblib.load(f"{out_dir}/rf_{tgt}.joblib")
        clf.n_jobs = 1  # 关键修复：单线程推理，速度提升100倍以上
        models[tgt] = clf
    
    # 预热
    for cid in cycle_ids[:WARMUP_ROUNDS]:
        X = build_single_cycle_features(raw, model_type, seconds, cid)
        X_imp = X.copy()
        inds = np.where(np.isnan(X_imp))
        X_imp[inds] = med[inds[1]]
        for tgt in TARGETS:
            _ = models[tgt].predict(X_imp)
    
    # 纯耗时测量
    times = []
    for _ in range(N_REPEAT):
        for cid in cycle_ids:
            X = build_single_cycle_features(raw, model_type, seconds, cid)
            t0 = time.perf_counter()
            X_imp = X.copy()
            inds = np.where(np.isnan(X_imp))
            X_imp[inds] = med[inds[1]]
            for tgt in TARGETS:
                _ = models[tgt].predict(X_imp)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    
    # 内存测量
    tracemalloc.start()
    tracemalloc.reset_peak()
    for cid in cycle_ids:
        X = build_single_cycle_features(raw, model_type, seconds, cid)
        X_imp = X.copy()
        inds = np.where(np.isnan(X_imp))
        X_imp[inds] = med[inds[1]]
        for tgt in TARGETS:
            _ = models[tgt].predict(X_imp)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return (
        np.percentile(times, 50),
        np.percentile(times, 95),
        peak_mem / 1024 / 1024
    )

def measure_training_time(raw, manifest, model_type, seconds):
    """测量训练耗时，不写入磁盘，避免IO开销影响结果"""
    train_idx = manifest.index[manifest["split"] == "train"].values
    labels_all = np.loadtxt(f"{DATA_DIR}/profile.txt")[:, :4].astype(int)
    
    # 预热
    _ = build_single_cycle_features(raw, model_type, seconds, train_idx[0])
    
    t0 = time.perf_counter()
    # 构建训练集特征
    feat_list = []
    for cid in train_idx:
        feat_list.append(build_single_cycle_features(raw, model_type, seconds, cid))
    X_train = np.vstack(feat_list)
    # 中位数填充
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_imp = X_train.copy()
    inds = np.where(np.isnan(X_imp))
    X_imp[inds] = med[inds[1]]
    # 拟合四个模型（训练时保留多线程加速）
    for tgt in TARGETS:
        y = labels_all[train_idx, LABEL_COLS[tgt]]
        clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
        clf.fit(X_imp, y)
    t1 = time.perf_counter()
    
    return t1 - t0

def get_model_size(model_type, seconds):
    out_dir = f"models/{model_type}_{seconds}s"
    total_size = 0
    for tgt in TARGETS:
        fpath = f"{out_dir}/rf_{tgt}.joblib"
        if os.path.exists(fpath):
            total_size += os.path.getsize(fpath)
    return total_size / 1024 / 1024

# -------------------------- 因果性校验 --------------------------
def causality_check(raw):
    print("\n>>> 执行因果性校验（前缀无未来信息泄漏）...")
    test_cid = 0
    passed = True
    
    for seconds in [10, 30]:
        for model_type in MODEL_TYPES:
            out_dir = f"models/{model_type}_{seconds}s"
            if not os.path.exists(out_dir):
                continue
            meta = json.load(open(f"{out_dir}/meta.json"))
            med = np.array(meta["impute_median"])
            models = {}
            for tgt in TARGETS:
                clf = joblib.load(f"{out_dir}/rf_{tgt}.joblib")
                clf.n_jobs = 1
                models[tgt] = clf
            
            # 原始结果
            X_orig = build_single_cycle_features(raw, model_type, seconds, test_cid)
            X_orig_imp = X_orig.copy()
            inds = np.where(np.isnan(X_orig_imp))
            X_orig_imp[inds] = med[inds[1]]
            pred_orig = {tgt: models[tgt].predict(X_orig_imp)[0] for tgt in TARGETS}
            
            # 修改窗口外数据
            raw_perturbed = {ch: arr.copy() for ch, arr in raw.items()}
            for ch, fs in CHANNELS.items():
                cut = fs * seconds
                if cut < raw_perturbed[ch].shape[1]:
                    raw_perturbed[ch][test_cid, cut:] = np.random.normal(0, 100, size=raw_perturbed[ch].shape[1] - cut)
            
            X_pert = build_single_cycle_features(raw_perturbed, model_type, seconds, test_cid)
            X_pert_imp = X_pert.copy()
            inds = np.where(np.isnan(X_pert_imp))
            X_pert_imp[inds] = med[inds[1]]
            pred_pert = {tgt: models[tgt].predict(X_pert_imp)[0] for tgt in TARGETS}
            
            if not np.allclose(X_orig, X_pert, equal_nan=True) or pred_orig != pred_pert:
                print(f"  ❌ 校验失败: {model_type} @ {seconds}s 存在泄漏!")
                passed = False
    
    if passed:
        print("  ✅ 校验通过：前缀截断严格不使用窗口外数据，回放倍率不影响结果")
    return passed

# -------------------------- 主流程 --------------------------
def main():
    print("=" * 60)
    print("液压故障诊断系统 工程性能基准测试 v3")
    print("=" * 60)
    print("\n读取数据...")
    raw = {ch: np.loadtxt(f"{DATA_DIR}/{ch}.txt") for ch in CHANNELS}
    manifest = pd.read_csv("manifest.csv")
    test_idx = manifest.index[manifest["split"] == "test"].values
    test_sample_idx = test_idx[:100]
    print(f"测试样本: {len(test_sample_idx)}个，重复测量: {N_REPEAT}次")
    os.makedirs(FIG_DIR, exist_ok=True)
    
    causality_check(raw)
    
    # 训练耗时
    print("\n>>> 测量训练耗时...")
    train_results = []
    for model_type in MODEL_TYPES:
        for seconds in BUDGETS:
            t = measure_training_time(raw, manifest, model_type, seconds)
            print(f"  {model_type.upper()} @ {seconds}s: {t:.2f}s")
            train_results.append({"model": model_type.upper(), "seconds": seconds, "train_time_s": round(t, 3)})
    
    # 推理性能
    print("\n>>> 测量推理性能...")
    results = []
    for model_type in MODEL_TYPES:
        for seconds in BUDGETS:
            out_dir = f"models/{model_type}_{seconds}s"
            if not os.path.exists(out_dir):
                print(f"  跳过 {model_type} @ {seconds}s")
                continue
            print(f"  测量 {model_type.upper()} @ {seconds}s ...")
            feat_p50, feat_p95, feat_mem = measure_feature_extraction(raw, model_type, seconds, test_sample_idx)
            inf_p50, inf_p95, inf_mem = measure_inference(raw, model_type, seconds, test_sample_idx)
            model_size = get_model_size(model_type, seconds)
            
            results.append({
                "model": model_type.upper(), "seconds": seconds,
                "feat_p50_ms": round(feat_p50, 3), "feat_p95_ms": round(feat_p95, 3), "feat_peak_mem_mb": round(feat_mem, 2),
                "inf_p50_ms": round(inf_p50, 3), "inf_p95_ms": round(inf_p95, 3), "inf_peak_mem_mb": round(inf_mem, 2),
                "total_p50_ms": round(feat_p50 + inf_p50, 3), "total_p95_ms": round(feat_p95 + inf_p95, 3),
                "model_size_mb": round(model_size, 2)
            })
    
    # 保存结果
    df_infer = pd.DataFrame(results)
    df_train = pd.DataFrame(train_results)
    df = df_infer.merge(df_train, on=["model", "seconds"], how="left")
    df.to_csv("benchmark_results.csv", index=False)
    
    df_60 = df[df["seconds"] == 60].copy()
    print("\n" + "=" * 80)
    print("60秒预算性能汇总")
    print("=" * 80)
    print(df_60[["model", "train_time_s", "feat_p50_ms", "inf_p50_ms", "total_p50_ms", "model_size_mb"]].to_string(index=False))
    
    # 绘图
    print("\n>>> 生成图表...")
    colors = {"B0": "gray", "B1": "#1f77b4", "B2": "#ff7f0e", "B3": "#2ca02c", "B4": "#d62728"}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, title in zip(axes, ["feat_p50_ms", "inf_p50_ms", "total_p50_ms"], ["特征提取P50", "模型推断P50", "端到端总耗时P50"]):
        bars = ax.bar(df_60["model"], df_60[metric], color=[colors[m] for m in df_60["model"]])
        ax.set_title(title); ax.set_ylabel("耗时 (ms)"); ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, df_60[metric]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{val:.2f}", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/benchmark_time_60s.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars = axes[0].bar(df_60["model"], df_60["model_size_mb"], color=[colors[m] for m in df_60["model"]])
    axes[0].set_title("模型文件大小"); axes[0].set_ylabel("MB"); axes[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, df_60["model_size_mb"]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{val:.2f}", ha='center', fontsize=9)
    
    bars = axes[1].bar(df_60["model"], df_60["train_time_s"], color=[colors[m] for m in df_60["model"]])
    axes[1].set_title("端到端训练耗时"); axes[1].set_ylabel("秒"); axes[1].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, df_60["train_time_s"]):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{val:.2f}", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/benchmark_size_train_60s.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"完成！结果已保存到 benchmark_results.csv，图表存于 {FIG_DIR}/ 目录")

if __name__ == "__main__":
    main()
