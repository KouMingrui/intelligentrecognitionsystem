# 液压系统状态监测与智能故障诊断系统

## 项目概述

本项目基于 **UCI 液压系统状态监测数据集**，实现了一套完整的液压系统状态监测与智能故障诊断系统。系统通过采集14个传感器的信号数据，对液压系统的四个关键部件进行实时故障诊断和预测。

> **配置说明**：本项目采用独立脚本等价实现（等价于 `train.py --config`），每个方案对应独立的训练脚本，结构清晰，便于理解和调试。

## 数据准备

### 1. 数据获取

从 [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems) 下载原始数据集，解压后将所有 `.txt` 文件放入 `condition+monitoring+of+hydraulic+systems/` 目录。

目录结构应包含以下文件：

```
condition+monitoring+of+hydraulic+systems/
├── PS1.txt ~ PS6.txt    # 压力传感器数据
├── EPS1.txt             # 功率传感器数据
├── FS1.txt, FS2.txt     # 流量传感器数据
├── TS1.txt ~ TS4.txt    # 温度传感器数据
├── VS1.txt              # 振动传感器数据
└── profile.txt          # 标签文件
```

### 2. 数据划分

运行数据划分脚本，生成训练/验证/测试集划分清单：

```bash
python prepare.py
```

执行后会生成 `manifest.csv` 文件，包含每个样本的数据集划分（train/val/test）。

## 数据说明

### 传感器通道（14个）

| 通道 | 采样频率 | 描述 |
|------|----------|------|
| PS1-PS6 | 100 Hz | 压力传感器 |
| EPS1 | 100 Hz | 功率传感器 |
| FS1, FS2 | 10 Hz | 流量传感器 |
| TS1-TS4 | 1 Hz | 温度传感器 |
| VS1 | 1 Hz | 振动传感器 |

### 监测目标（4个部件）

- **cooler** (冷却器): 健康等级 20-100
- **valve** (阀门): 健康等级 0-100
- **pump** (泵): 健康等级 0-100
- **accumulator** (蓄能器): 健康等级 0-180

### 正常状态基准值

| 部件 | 正常值 |
|------|--------|
| cooler | 100 |
| valve | 100 |
| pump | 0 |
| accumulator | 130 |

## 项目结构

```
.
├── prepare.py               # 数据划分脚本
├── train_b0.py              # 方案B0训练脚本（基础统计特征）
├── train_b1.py              # 方案B1训练脚本（分段统计+差分特征）
├── train_b2.py              # 方案B2训练脚本（分段统计特征，消融差分）
├── train_b3.py              # 方案B3训练脚本（B1特征+物理交叉特征）
├── train_b4.py              # 方案B4训练脚本（B0特征+物理交叉特征）
├── evaluate.py              # 模型性能评估脚本
├── alarm_evaluate.py        # 预测与告警生成脚本
├── evaluate_robustness.py   # 模型鲁棒性评估脚本（Macro-F1/异常定位/告警）
├── replay.py                # 可视化回放脚本
├── benchmark.py             # 性能基准测试脚本
├── plot_confusion.py        # 混淆矩阵可视化脚本
├── plot_one_cycle.py        # 单周期错误案例分析脚本
├── plot_robustness.py       # 鲁棒性测试结果可视化脚本
├── extract_all_errors.py    # 提取所有错误预测案例脚本
│
├── models/                  # 模型存储目录
│   ├── b0_10s/ ~ b0_60s/    # B0方案模型
│   ├── b1_10s/ ~ b1_60s/    # B1方案模型
│   ├── b2_10s/ ~ b2_60s/    # B2方案模型
│   ├── b3_10s/ ~ b3_60s/    # B3方案模型
│   └── b4_10s/ ~ b4_60s/    # B4方案模型
│
├── figures/                                # 图表输出目录
├── condition+monitoring+of+hydraulic+systems/  # 原始数据目录
├── manifest.csv             # 数据集划分清单
├── environment.txt          # 依赖环境配置
└── *.csv/json               # 结果输出文件
```

## 特征工程

### 方案B0：基础统计特征

每个通道提取 **6个统计量** + **1个近常数标记**:

- 均值 (mean)
- 标准差 (std)
- 最小值 (min)
- 最大值 (max)
- 均方根 (rms)
- 线性趋势斜率 (slope)
- 近常数标记 (missing)

### 方案B1：分段统计+差分特征

在B0基础上增加 **5个特征**:

- 三段均值 (s1_mean, s2_mean, s3_mean)
- 段间差分 (diff_1_2, diff_2_3)

### 方案B2：消融差分特征

在B0基础上仅增加 **3个分段均值**（去掉差分特征）

### 方案B3：B1特征+物理交叉特征

在B1基础上增加 **2个物理交叉特征**:

- 能流比 (energy_per_flow) = EPS1均值 / (FS1均值 + 1e-5)
- 温差 (temp_diff) = TS1均值 - TS4均值

### 方案B4：B0特征+物理交叉特征

在B0基础上增加 **2个物理交叉特征**:

- 能流比 (energy_per_flow) = EPS1均值 / (FS1均值 + 1e-5)
- 温差 (temp_diff) = TS1均值 - TS4均值

## 快速开始

### 1. 环境配置

```bash
conda env create -f environment.yml  # 或使用 environment.txt 中的依赖
conda activate hydraulic
```

### 2. 数据准备

```bash
# 下载数据并放入 condition+monitoring+of+hydraulic+systems/ 目录后，执行数据划分
python prepare.py
```

### 3. 模型训练

```bash
# 方案B0训练（基础统计特征）
python train_b0.py

# 方案B1训练（分段统计+差分特征）
python train_b1.py

# 方案B2训练（分段统计特征，消融差分）
python train_b2.py

# 方案B3训练（B1特征+物理交叉特征）
python train_b3.py

# 方案B4训练（B0特征+物理交叉特征）
python train_b4.py
```

### 4. 生成预测与告警

```bash
# 生成B0模型的预测和告警
python alarm_evaluate.py --model b0

# 生成B1模型的预测和告警
python alarm_evaluate.py --model b1

# 生成B2模型的预测和告警
python alarm_evaluate.py --model b2

# 生成B3模型的预测和告警
python alarm_evaluate.py --model b3

# 生成B4模型的预测和告警
python alarm_evaluate.py --model b4
```

### 5. 模型评估

```bash
# 生成完整的评测指标
python evaluate.py

# 测试模型鲁棒性（输出 Macro-F1、异常定位 F1 与告警 F1）
python evaluate_robustness.py

# 性能基准测试（耗时、内存、模型大小）
python benchmark.py
```

### 6. 可视化

```bash
# 回放指定周期的预测过程
python replay.py --cycle 100 --model b1 --speed normal

# 生成混淆矩阵可视化（20张图）
python plot_confusion.py

# 绘制鲁棒性测试结果对比图
python plot_robustness.py

# 分析单周期错误案例
python plot_one_cycle.py

# 提取所有错误预测案例
python extract_all_errors.py
```


## 评测指标

### 分类性能

- **Macro-F1**: 所有类别的F1分数平均值
- **Recall per Class**: 每个健康等级的召回率
- **Strict Joint Accuracy**: 四部件全部预测正确的比例
- **Anomaly Localization F1**: 异常定位F1（二分类：是否异常）

### 稳定性分层

- **Stable Flag 0/1 Joint Accuracy**: 按稳定性标记分层的严格联合准确率

### 告警性能

- **Alarm F1**: 异常检测的F1分数
- **响应时间**: 告警触发的延迟（30秒内/60秒内/漏报数）

### 鲁棒性测试

测试条件包括：

- 干净数据 (clean)
- 高斯噪声扰动 (noise_0.05, noise_0.10)
- 传感器缺失 (mask_FS1, mask_FS1_FS2)

在五种条件下分别报告四部件平均与逐部件 Macro-F1，以及异常定位 F1 与告警 F1。其中加噪场景性能衰减温和，传感器缺失（尤其屏蔽 FS1）会使泵的诊断性能大幅下降。

### 性能基准测试

测量指标（同机比较，不做跨机器排名）：

- **特征提取耗时 P50/P95**: 单周期从原始信号提取特征
- **模型推断耗时 P50/P95**: 单周期预测4个部件
- **内存峰值**: 特征提取+推断过程的峰值内存
- **模型文件大小**: 4个部件模型总大小

## 输出文件

| 文件 | 描述 |
|------|------|
| `b*_summary.csv` | 各方案的训练/验证/测试集 Macro-F1 汇总 |
| `predictions_*.csv` | 所有测试样本的预测结果 |
| `alarms_*.json` | 触发的告警记录 |
| `metrics.json` | 完整评测指标（含稳定性分层） |
| `robustness_summary.csv` | 鲁棒性测试结果（逐部件 Macro-F1） |
| `robustness_localization.csv` | 五种扰动条件下的异常定位 P/R/F1（逐观测预算） |
| `robustness_alarm.csv` | 五种扰动条件下的告警 P/R/F1 与健康部件误报率 |
| `benchmark_results.csv` | 性能基准测试结果 |
| `figures/cm_*.png` | 混淆矩阵可视化图（20张） |
| `figures/robustness_*_60s.png` | 鲁棒性测试结果对比图（4张） |
| `figures/analysis_cycle_*.png` | 单周期错误案例分析图（5张） |
| `figures/benchmark_*.png` | 性能基准测试图表 |


## 参考

- 数据集来源: [UCI Machine Learning Repository - Condition Monitoring of Hydraulic Systems](https://doi.org/10.24432/C5CW21)
- 算法: RandomForestClassifier (n_estimators=200)
- 随机种子: 2026
