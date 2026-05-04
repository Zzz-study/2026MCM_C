# MCM Problem C - 舞蹈比赛分析系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MCM](https://img.shields.io/badge/MCM-Problem%20C-red)](https://www.comap.com/)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)

本项目是Mathematical Contest in Modeling (MCM) Problem C "Dancing with the Stars" 的完整解决方案。项目包含四个子模块，分别解决舞蹈比赛投票系统的不同问题。

---

## 目录

- [项目概述](#项目概述)
- [问题一：粉丝投票估计系统](#问题一粉丝投票估计系统)
- [问题二：投票机制比较分析](#问题二投票机制比较分析)
- [问题三：双分支神经网络分析](#问题三双分支神经网络分析)
- [问题四：舞蹈进化生态系统模型](#问题四舞蹈进化生态系统模型)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [依赖安装](#依赖安装)

---

## 项目概述

本项目对"与星共舞"舞蹈比赛的投票系统进行全面的量化分析，包括：

- **粉丝投票估计**：通过混合建模（整数规划+贝叶斯MCMC）从公开数据中反推粉丝投票
- **投票机制比较**：对比排名法与百分比法的公平性和偏向性
- **特征影响分析**：使用双分支神经网络+SHAP解释分析选手特征对评委和粉丝的影响
- **优化投票系统**：基于进化算法构建更公平、更娱乐的新投票模型

---

## 问题一：粉丝投票估计系统

**目录**: [`Q1_final/`](Q1_final/)

### 核心功能

| 模块 | 算法 | 说明 |
|------|------|------|
| 排名法建模 | 整数规划 (PuLP) | 求解粉丝排名，满足淘汰一致性约束 |
| 百分比法建模 | 贝叶斯MCMC (PyMC3) | 估计粉丝投票百分比及置信区间 |
| 动态特征提取 | HMM (隐马尔可夫模型) | 捕捉选手人气时变效应 |
| 特征重要性分析 | 随机森林 | 量化各特征对投票预测的影响 |
| 模型验证 | 灵敏度分析 | 测试模型稳健性 |

### 主要输出

- `fan_vote_estimates.csv` - 粉丝投票估计结果
- `feature_importance.png` - 特征重要性可视化
- `uncertainty_distribution.png` - 投票不确定性分布
- `sensitivity_analysis.png` - 灵敏度分析结果

### 快速运行

```bash
cd Q1_final
pip install -r requirements.txt
python dance_competition_analysis.py
```

---

## 问题二：投票机制比较分析

**目录**: [`Q2_ver1/`](Q2_ver1/)

### 核心功能

| 分析维度 | 核心指标 | 说明 |
|----------|----------|------|
| 宏观比较 | ρ (Spearman)、W (权重)、D (偏差) | 量化粉丝偏向性 |
| 微观分析 | ΔRank_rel (排名差异)、γ (挽救效应) | 分析争议选手 |
| 规则模拟 | 淘汰规则扩展 | 模拟"评委选择淘汰"规则 |
| 特殊分析 | 28季纵向/横向比较 | 对比新旧机制 |

### 核心指标

- **粉丝相关性系数 (ρ)**: Spearman秩相关系数，越接近1越偏向粉丝
- **粉丝贡献权重 (W)**: 回归反推的粉丝投票实际贡献占比
- **纯粉丝偏差 (D)**: Kendall tau距离，越小越接近纯粉丝排名
- **排名相对差异 (ΔRank_rel)**: |Rank_R - Rank_P| / n × 100%

### 快速运行

```bash
cd Q2_ver1/dance_analysis
python main.py
```

---

## 问题三：双分支神经网络分析

**目录**: [`Q3_ver1/`](Q3_ver1/)

### 核心功能

| 模块 | 技术 | 说明 |
|------|------|------|
| 双分支神经网络 | PyTorch | 同时预测评委评分和粉丝投票 |
| 多任务学习 | 加权损失函数 | 优化两个任务的联合表现 |
| 超参数优化 | Optuna | 自动搜索最佳网络结构和训练参数 |
| 模型解释 | SHAP | 量化特征对两个目标的影响差异 |
| 基准对比 | XGBoost | 验证多任务学习的优势 |

### 模型架构

```
输入层
  ↓
共享特征提取层 (128-256神经元)
  ↓
┌───────────────┴───────────────┐
↓                               ↓
评委评分分支                  粉丝投票分支
(32-64神经元)                 (32-64神经元)
↓                               ↓
评委评分预测                 粉丝投票预测
```

### 快速运行

```bash
cd Q3_ver1
pip install -r requirements.txt
python dual_branch_analysis.py
```

---

## 问题四：舞蹈进化生态系统模型

**目录**: [`Q4_ver1/`](Q4_ver1/)

### 核心概念

将舞蹈比赛建模为**进化生态系统**，每个选手具有：

| 特征 | 生物学类比 | 计算方法 |
|------|-----------|----------|
| 技术基因评分 | 基因型 | 多维度评委评分加权 |
| 环境适应性 | 适应度 | 标准化粉丝投票 |
| 进化趋势 | 达尔文适应度 | 进步速度 + λ·创新性 |
| 创新性 | 变异 | 评委评分标准差 |
| 社交邻域 | 共生关系 | 行业/舞伴网络 |

### 可调节参数

- **专业-娱乐平衡滑块**: 调节技术分与粉丝分的权重
- **悬念强度滑块**: 控制淘汰的可预测性
- **随机事件开关**: 启用/禁用可控随机事件

### 预设模式

| 模式 | 平衡滑块 | 悬念滑块 | 特点 |
|------|----------|----------|------|
| 专业优先 | 0.2 | 0.3 | 重技术、低悬念 |
| 娱乐优先 | 0.8 | 0.7 | 重人气、高悬念 |
| 平衡模式 | 0.5 | 0.5 | 均衡配置 |
| 高悬念 | 0.5 | 0.9 | 最大化娱乐性 |

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/MCM_C_DanceCompetition.git
cd MCM_C_DanceCompetition
```

### 2. 安装依赖

各问题依赖可单独安装：

```bash
# 问题一（最完整）
pip install -r Q1_final/requirements.txt

# 问题二
pip install -r Q2_ver1/requirements.txt

# 问题三（需要GPU）
pip install -r Q3_ver1/requirements.txt
```

### 3. 运行分析

```bash
# 运行问题一（完整分析，约30-60分钟）
cd Q1_final
python dance_competition_analysis.py

# 运行快速测试（5-10秒）
python dance_competition_analysis_quick_test.py

# 运行问题二
cd ../Q2_ver1/dance_analysis
python main.py

# 运行问题三
cd ../../Q3_ver1
python dual_branch_analysis.py
```

---

## 项目结构

```
MCM_C_Final/
├── Q1_final/                          # 问题一：粉丝投票估计
│   ├── dance_competition_analysis.py      # 主分析脚本
│   ├── dance_competition_analysis_quick_test.py  # 快速测试
│   ├── dance_competition_final_processed.csv    # 输入数据
│   ├── fan_vote_constraints.json            # 约束数据
│   ├── requirements.txt                     # 依赖列表
│   ├── README.md                           # 详细说明
│   ├── QUICKSTART.txt                      # 快速参考
│   └── 使用指南.md                         # 中文使用指南
│
├── Q2_ver1/                           # 问题二：投票机制比较
│   ├── dance_analysis/                    # 分析模块包
│   │   ├── __init__.py
│   │   ├── data_loader.py                 # 数据加载
│   │   ├── macro_comparison.py            # 宏观比较
│   │   ├── micro_analysis.py              # 微观分析
│   │   ├── rule_simulation.py             # 规则模拟
│   │   ├── season28_analysis.py           # 28季分析
│   │   ├── visualization.py               # 可视化
│   │   └── main.py                        # 主程序
│   ├── 输出/                              # 输出目录
│   │   ├── 表格输出/                      # CSV结果
│   │   ├── 可视化/                        # PNG图表
│   │   └── 分析报告/                      # Markdown报告
│   ├── fan_vote_estimates.csv            # 输入数据
│   ├── requirements.txt
│   └── README.md
│
├── Q3_ver1/                           # 问题三：双分支神经网络
│   ├── dual_branch_analysis.py            # 主分析脚本
│   ├── models/                            # 保存的模型
│   ├── outputs/                           # 输出目录
│   │   ├── metrics/                       # 性能指标
│   │   ├── shap_results/                  # SHAP解释
│   │   └── controversial_cases/           # 争议案例分析
│   ├── Q2_data/                          # Q2输入数据
│   ├── requirements.txt
│   └── prompt.txt                         # 技术规范
│
├── Q4_ver1/                           # 问题四：进化生态系统
│   ├── test1.py                          # 主分析脚本
│   ├── dance_competition_features.csv    # 特征数据
│   ├── fan_vote_estimates.xlsx           # 粉丝投票估计
│   └── prompt.txt                        # 技术规范
│
└── README.md                          # 本文件
```

---

## 依赖安装

### Python版本

- Python 3.8+

### 完整依赖列表

| 包名 | 版本 | 用途 |
|------|------|------|
| pandas | >=1.5.0 | 数据处理 |
| numpy | >=1.23.0 | 数值计算 |
| scipy | >=1.9.0 | 科学计算 |
| scikit-learn | >=1.1.0 | 机器学习 |
| matplotlib | >=3.6.0 | 可视化 |
| seaborn | >=0.12.0 | 高级可视化 |
| PuLP | >=2.7.0 | 整数规划 |
| pymc3 | >=3.11.5 | 贝叶斯建模 |
| arviz | >=0.15.0 | 贝叶斯可视化 |
| hmmlearn | >=0.3.0 | 隐马尔可夫模型 |
| torch | >=1.12.0 | 深度学习 |
| shap | >=0.41.0 | 模型解释 |
| optuna | >=3.0.5 | 超参数优化 |
| xgboost | >=1.6.2 | 基准模型 |
| tqdm | >=4.65.0 | 进度条 |

### 一键安装所有依赖

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn tqdm
pip install PuLP pymc3 arviz hmmlearn
pip install torch torchvision shap optuna xgboost
```


---

## 技术亮点

- **混合建模**: 整数规划 + 贝叶斯MCMC + HMM 三重建模
- **多任务学习**: 双分支神经网络同时预测两个目标
- **可解释AI**: SHAP值量化特征影响，提供业务洞见
- **进化算法**: 模拟生物进化，构建创新的投票系统
- **完整的可视化**: 从数据探索到结果展示的全流程图表

---

## 作者

- **团队**: MCM Problem C Team
- **年份**: 2024-2025

---

## 许可

本项目仅供学术研究和教育使用。

---

## 联系方式

如有问题，请通过GitHub Issues提交。
