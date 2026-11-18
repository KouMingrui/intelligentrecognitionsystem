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
├── evaluate.py              # 模型性能评估脚本
├── alarm_evaluate.py        # 预测与告警生成脚本
├── evaluate_robustness.py   # 模型鲁棒性评估脚本
├── replay.py                # 可视化回放脚本
│
├── models/                  # 模型存储目录
│   ├── b0_10s/              # B0方案-10秒观测模型
│   ├── b0_30s/
│   ├── b0_60s/
│   ├── b1_10s/              # B1方案-10秒观测模型
│   ├── b1_30s/
│   ├── b1_60s/
│   ├── b2_10s/              # B2方案-10秒观测模型
│   ├── b2_30s/
│   └── b2_60s/
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
# 方案B0训练
python train_b0.py

# 方案B1训练
python train_b1.py

# 方案B2训练
python train_b2.py
```

### 4. 生成预测与告警

```bash
# 生成B0模型的预测和告警
python alarm_evaluate.py --model b0

# 生成B1模型的预测和告警
python alarm_evaluate.py --model b1

# 生成B2模型的预测和告警
python alarm_evaluate.py --model b2
```

### 5. 模型评估

```bash
# 生成完整的评测指标
python evaluate.py

# 测试模型鲁棒性
python evaluate_robustness.py
```

### 6. 可视化回放

```bash
# 回放指定周期的预测过程
python replay.py --cycle 100 --model b1 --speed normal
```

## 评测指标

### 分类性能

- **Macro-F1**: 所有类别的F1分数平均值
- **Recall per Class**: 每个健康等级的召回率
- **Strict Joint Accuracy**: 四部件全部预测正确的比例

### 告警性能

- **Alarm F1**: 异常检测的F1分数
- **响应时间**: 告警触发的延迟（30秒内/60秒内）

### 鲁棒性测试

测试条件包括：

- 干净数据 (clean)
- 高斯噪声扰动 (noise_0.05, noise_0.10)
- 传感器缺失 (mask_FS1, mask_FS1_FS2)

## 输出文件

| 文件 | 描述 |
|------|------|
| `b*_summary.csv` | 各方案的训练/验证/测试集 Macro-F1 汇总 |
| `predictions_*.csv` | 所有测试样本的预测结果 |
| `alarms_*.json` | 触发的告警记录 |
| `metrics.json` | 完整评测指标 |
| `robustness_summary.csv` | 鲁棒性测试结果 |


## 参考

- 数据集来源: [UCI Machine Learning Repository - Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems)
- 算法: RandomForestClassifier (n_estimators=200)
- 随机种子: 2026
