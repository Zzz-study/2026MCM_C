# 舞蹈比赛粉丝投票分析系统

## 项目概述
本项目实现了一个完整的舞蹈比赛粉丝投票估计系统，包括：
- 排名法建模（整数规划）
- 百分比法建模（贝叶斯-MCMC）
- HMM动态特征提取
- 随机森林特征重要性分析
- 模型验证与灵敏度分析

## 文件说明
- `dance_competition_analysis.py` - 主分析脚本（1079行完整实现）
- `test_data_load.py` - 数据加载测试脚本
- `requirements.txt` - Python依赖包列表
- `dance_competition_final_processed.csv` - 最终样本长表
- `fan_vote_constraints.json` - 粉丝投票约束数据
- `dance_competition_features.csv` - 特征工程中间表

## 安装依赖

### 方法1: 使用pip安装（推荐）
```bash
pip install -r requirements.txt
```

### 方法2: 手动安装主要依赖
```bash
pip install pandas numpy tqdm
pip install pulp  # 整数规划
pip install pymc3 arviz  # 贝叶斯建模
pip install hmmlearn  # 隐马尔可夫模型
pip install scikit-learn  # 随机森林
pip install matplotlib seaborn  # 可视化
```

## 运行步骤

### 步骤1: 测试数据加载
```bash
cd "d:\Mathematical modeling\MCM_C\Q1_final"
python test_data_load.py
```

如果数据加载成功，会显示：
- ✓ CSV文件行数、列数
- ✓ JSON文件记录数
- ✓ 所有必需列都存在

### 步骤2: 运行完整分析
```bash
python dance_competition_analysis.py
```

**注意**: 完整运行可能需要30-60分钟，具体取决于：
- 数据规模
- 贝叶斯MCMC采样次数（默认2000次）
- CPU核心数

### 步骤3: 查看结果

运行完成后，会生成以下文件：

**CSV结果文件**:
- `fan_vote_estimates.csv` - 粉丝投票估计结果（含排名法和百分比法）
- `elimination_consistency.csv` - 淘汰结果一致性分析
- `vote_uncertainty.csv` - 投票不确定性度量
- `sensitivity_cv.csv` - 灵敏度分析变异系数
- `feature_importance.csv` - 特征重要性分析

**可视化图表**:
- `feature_importance.png` - 特征重要性对比图
- `uncertainty_distribution.png` - 不确定性分布直方图
- `sensitivity_analysis.png` - 灵敏度分析箱线图
- `fan_vote_ci_example.png` - 粉丝投票置信区间示例

## 代码结构

### 1. 数据加载与预处理（第50-180行）
- `load_data()` - 加载3个数据文件并合并
- `extract_features()` - 提取建模特征

### 2. 排名法建模（第182-320行）
- `rank_based_vote_estimation()` - 整数规划求解粉丝排名
- 使用PuLP库实现线性优化
- 约束条件：排名唯一性 + 淘汰一致性

### 3. 百分比法建模（第322-480行）
- `percent_based_vote_estimation()` - 贝叶斯模型估计粉丝百分比
- 使用PyMC3实现MCMC采样
- 输出后验均值和95%置信区间

### 4. HMM动态特征（第482-580行）
- `add_hmm_dynamic_features()` - 捕捉选手人气时变效应
- 使用hmmlearn库，3个隐藏状态（低/中/高人气）

### 5. 随机森林分析（第582-700行）
- `random_forest_stratified_model()` - 预测投票并分析特征重要性
- 回归模型：预测投票数
- 分类模型：预测淘汰概率

### 6. 模型验证（第702-850行）
- `model_validation()` - 一致性度量、不确定性评估

### 7. 灵敏度分析（第852-950行）
- `sensitivity_analysis()` - 改变总票数假设，测试稳定性

### 8. 结果输出（第952-1050行）
- `save_and_visualize_results()` - 保存CSV和生成图表

## 常见问题

### Q1: 运行时出现编码错误
**解决方案**: 
```python
# 修改load_data函数，添加encoding参数
final_df = pd.read_csv("dance_competition_final_processed.csv", encoding='gbk')
```

### Q2: PyMC3采样过慢
**解决方案**:
- 减少采样次数：`pm.sample(1000, tune=500)` 改为 `pm.sample(500, tune=250)`
- 使用单核：`cores=1`

### Q3: 整数规划求解失败
**可能原因**:
- 约束过于严格（无可行解）
- 需要安装CBC求解器：`conda install -c conda-forge coincbc`

### Q4: HMM建模失败
**解决方案**:
- 参赛周数过少（<3周）会自动跳过
- 减少隐藏状态数：`n_components=2`

### Q5: 内存不足
**解决方案**:
- 分批处理赛季：修改groupby逻辑，每次只处理1-2个赛季
- 减少贝叶斯采样次数

## 输出结果解读

### fan_vote_estimates.csv
包含每个选手-周的投票估计：
- **排名法**: `fan_rank`（粉丝排名）, `fan_votes`（投票数）
- **百分比法**: `fan_percent_mean`（百分比均值）, `fan_votes_mean`（投票数均值）, `fan_percent_95ci_lower/upper`（置信区间）

### elimination_consistency.csv
每周的淘汰预测一致性：
- `consistency=1`: 预测正确
- `consistency=0`: 预测错误
- 整体一致性 = 正确周数 / 总周数

### feature_importance.csv
特征对投票预测的影响力排序：
- `vote_pred_importance`: 对投票数预测的重要性
- `elimination_pred_importance`: 对淘汰预测的重要性

## 模型核心假设

1. **排名法**: 
   - 粉丝排名 + 评委排名 = 总排名
   - 总排名最高者被淘汰

2. **百分比法**:
   - 粉丝百分比 + 评委百分比 = 总百分比
   - 总百分比最低者被淘汰

3. **总票数假设**: 默认每周10000票（可调整）

4. **人气状态**: HMM假设3个隐藏状态（低/中/高人气）

## 性能优化建议

如需加速运行：
1. 使用多核并行：`cores=2` 或 `cores=4`（贝叶斯采样）
2. 减少MCMC采样：`pm.sample(500, tune=250)`
3. 降低随机森林树数：`n_estimators=50`
4. 跳过HMM建模（注释相关代码）

## 联系方式
如有问题，请检查：
1. 数据文件格式是否正确
2. Python版本 >= 3.8
3. 所有依赖包是否安装成功

## 版本信息
- Version: 1.0
- Date: 2024
- Python: 3.8+
