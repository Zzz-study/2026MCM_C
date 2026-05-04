# 舞蹈比赛投票机制分析系统

## 项目概述

本项目对舞蹈比赛的两种投票组合方法（排名法 vs 百分比法）进行全面的量化比较分析，评估不同机制的粉丝偏向性。

## 核心功能

### 1. 宏观比较分析
- 计算粉丝相关性系数 (ρ) - Spearman秩相关
- 计算粉丝贡献权重 (W) - 回归反推
- 计算纯粉丝偏差 (D) - Kendall tau距离

### 2. 争议选手微观分析
- 分析特定争议选手在不同机制下的表现差异
- 计算排名相对差异 (ΔRank_rel)
- 计算粉丝挽救效应系数 (γ)

### 3. 规则扩展分析
- 模拟"评委选择淘汰"规则的影响
- 评估规则对两种方法差异的影响

### 4. 28季特殊分析
- 纵向比较：28季 vs 前2季（排名法）
- 横向比较：28季 vs 3-27季（百分比法）

## 项目结构

```
MCM_C/Q2_ver1/
├── dance_analysis/              # 分析模块包
│   ├── __init__.py             # 包初始化
│   ├── data_loader.py          # 数据加载与预处理
│   ├── macro_comparison.py     # 宏观比较分析
│   ├── micro_analysis.py       # 争议选手分析
│   ├── rule_simulation.py      # 规则模拟
│   ├── season28_analysis.py    # 28季特殊分析
│   ├── visualization.py        # 可视化生成
│   └── main.py                 # 主程序入口
├── 输出/                        # 输出目录
│   ├── 表格输出/               # CSV结果文件
│   ├── 可视化/                 # PNG图表文件
│   └── 分析报告/               # Markdown报告
├── fan_vote_estimates.csv      # 输入数据
├── requirements.txt            # 依赖包列表
└── README.md                   # 本文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行完整分析

```bash
cd dance_analysis
python main.py
```

### 运行单个模块

```python
# 示例：仅运行宏观比较分析
from dance_analysis import load_fan_vote_estimates, preprocess_data, compare_all_seasons

df = load_fan_vote_estimates("fan_vote_estimates.csv")
df = preprocess_data(df)
results_df, summary_df = compare_all_seasons(df)
```

## 输出文件

### 表格输出
- `macro_comparison.csv`: 各赛季宏观指标
- `macro_summary.csv`: 跨赛季汇总统计
- `micro_analysis.csv`: 争议选手分析
- `rule_impact.csv`: 规则影响分析
- `season28_vertical.csv`: 28季纵向比较
- `season28_horizontal.csv`: 28季横向比较

### 可视化
- `fan_bias_comparison.png`: 粉丝偏向性对比图
- `controversial_cases_radar.png`: 争议选手雷达图
- `controversial_cases_scatter.png`: 争议选手散点图
- `rule_impact_heatmap.png`: 规则影响热力图
- `season28_comparison_bar.png`: 28季对比柱状图

### 分析报告
- `analysis_summary.md`: 完整分析报告（包含执行摘要、方法论、结果和推荐）

## 核心指标说明

### 粉丝相关性系数 (ρ)
- **定义**: Spearman秩相关系数
- **范围**: [-1, 1]
- **解读**: 越接近1，越偏向粉丝

### 粉丝贡献权重 (W)
- **定义**: 通过线性回归反推的粉丝投票实际贡献占比
- **范围**: [0, 1]
- **解读**: 数值越大，粉丝影响力越大

### 纯粉丝偏差 (D)
- **定义**: 基于Kendall tau距离
- **范围**: [0, 1]
- **解读**: 数值越小，越接近纯粉丝排名

### 排名相对差异 (ΔRank_rel)
- **定义**: |Rank_R - Rank_P| / n × 100%
- **解读**: <20%轻微差异，≥40%显著差异

### 粉丝挽救效应系数 (γ)
- **定义**: (Rank_Jonly - Rank_method) / Rank_Jonly × 100%
- **解读**: 正值越高，粉丝挽救效应越强

## 技术要点

1. **数据一致性**: 所有排名计算使用相同的数据源
2. **缺失值处理**: 对于粉丝百分比缺失的记录，使用排名法数据估算
3. **标准化**: 所有百分比数据标准化到0-100范围
4. **稳健性**: 回归模型添加截距项，检查多重共线性
5. **可视化**: 使用一致的颜色方案（排名法-蓝色，百分比法-橙色）

## 注意事项

1. 确保输入数据文件 `fan_vote_estimates.csv` 在正确位置
2. 运行前检查Python版本（建议3.8+）
3. 所有图表保存为300dpi PNG格式
4. 中文字符使用SimHei或Microsoft YaHei字体

## 版本信息

- **版本**: 1.0.0
- **最后更新**: 2026-02-01

## 许可证

本项目仅供学术研究使用。
