"""
舞蹈比赛粉丝投票分析系统
完整实现：排名法/百分比法建模、HMM动态特征、随机森林分析
"""

# ============================================================
# 一、环境准备与库导入
# ============================================================

# 基础数据处理
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 优化建模（排名法）
from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, lpSum, LpStatus, value

# 贝叶斯建模（百分比法）
import pymc as pm
import arviz as az
from scipy.stats import dirichlet

# 动态因素（HMM）
from hmmlearn import hmm

# 分层建模（随机森林）
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, mean_squared_error, classification_report, confusion_matrix

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

# 交叉验证
from sklearn.model_selection import LeaveOneGroupOut

# 设置随机种子
np.random.seed(42)


# ============================================================
# 二、数据加载与预处理
# ============================================================

def load_data(data_dir="./"):
    """
    加载所有数据文件并进行预处理
    
    Parameters:
    -----------
    data_dir : str
        数据文件所在目录
    
    Returns:
    --------
    final_df : DataFrame
        最终样本长表
    valid_df : DataFrame
        有效建模样本
    constraints_df : DataFrame
        投票约束表
    features_df : DataFrame
        特征工程表
    """
    print("=" * 60)
    print("步骤1: 加载数据文件")
    print("=" * 60)
    
    try:
        # 1. 加载最终样本长表（核心建模数据）
        print("正在加载 dance_competition_final_processed.csv...")
        final_df = pd.read_csv(f"{data_dir}/dance_competition_final_processed.csv", encoding='utf-8', encoding_errors='ignore')
        print(f"✓ 加载成功，共 {len(final_df)} 条记录")
        
        # 2. 加载粉丝投票约束（用于约束条件构建）
        print("正在加载 fan_vote_constraints.json...")
        with open(f"{data_dir}/fan_vote_constraints.json", "r", encoding='utf-8') as f:
            vote_constraints = json.load(f)
        
        # 转换约束为DataFrame（方便关联）
        constraints_list = []
        for item in vote_constraints:
            contestant_names = item.get("contestant_list", [])
            eliminated_names = item.get("eliminated_list", [])
            judge_ranks = item.get("judge_ranks", {})
            judge_percents = item.get("judge_percents", {})
            
            df_item = pd.DataFrame({
                "season": item["season"],
                "week": item["week"],
                "num_contestants": item.get("num_contestants", len(contestant_names)),
                "num_eliminated": item.get("num_eliminated", len(eliminated_names)),
                "celebrity_name": contestant_names,
                "is_eliminated": [name in eliminated_names for name in contestant_names],
                "judge_rank": [judge_ranks.get(name, 0) for name in contestant_names],
                "judge_percent": [judge_percents.get(name, 0.0) for name in contestant_names]
            })
            constraints_list.append(df_item)
        constraints_df = pd.concat(constraints_list, ignore_index=True)
        print(f"✓ 加载成功，共 {len(constraints_df)} 条约束记录")
        
        # 3. 加载特征工程中间表（仅用于特征解释，可选）
        print("正在加载 dance_competition_features.csv...")
        features_df = pd.read_csv(f"{data_dir}/dance_competition_features.csv", encoding='utf-8', encoding_errors='ignore')
        print(f"✓ 加载成功，共 {len(features_df)} 条特征记录")
        
        # 关联最终表与约束表（通过season+week+celebrity_name）
        print("\n正在合并数据表...")
        merged_df = pd.merge(
            final_df, 
            constraints_df, 
            on=["season", "week", "celebrity_name"], 
            how="inner"
        )
        
        # 筛选有效样本（文档建模建议）
        valid_df = merged_df[
            (merged_df["in_competition"] == True) & 
            (merged_df["non_competition_week"] == False) & 
            (merged_df["week"] <= merged_df["week_eliminated"])  # 排除退出/淘汰后样本
        ].reset_index(drop=True)
        
        print(f"✓ 合并完成，有效建模样本: {len(valid_df)} 条")
        print(f"  - 赛季范围: {valid_df['season'].min()} - {valid_df['season'].max()}")
        print(f"  - 周次范围: {valid_df['week'].min()} - {valid_df['week'].max()}")
        print(f"  - 参赛选手数: {valid_df['celebrity_name'].nunique()}")
        
        return final_df, valid_df, constraints_df, features_df
    
    except FileNotFoundError as e:
        print(f"✗ 错误：找不到数据文件 - {e}")
        print("请确保以下文件存在于指定目录:")
        print("  1. dance_competition_final_processed.csv")
        print("  2. fan_vote_constraints.json")
        print("  3. dance_competition_features.csv")
        raise
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        raise


def extract_features(valid_df):
    """
    特征工程：提取建模所需输入特征
    
    Parameters:
    -----------
    valid_df : DataFrame
        有效样本数据
    
    Returns:
    --------
    model_df : DataFrame
        包含特征和目标变量的建模数据
    """
    print("\n" + "=" * 60)
    print("步骤2: 特征工程")
    print("=" * 60)
    
    # 核心输入特征：技术相关+人气相关
    feature_cols = [
        "relative_level",  # 技术相关
        "celebrity_industry_encoded",  # 人气相关（行业）
        "ballroom_partner_encoded",    # 人气相关（舞伴）
        "celebrity_age_during_season_bin_encoded",  # 人气相关（年龄组）
        "season_stage_encoded",        # 赛季阶段
        "all_star_season",             # 特殊赛季标记
        "controversial_contestant"     # 争议选手标记
    ]
    
    # 目标变量与约束相关列
    target_cols = [
        "eliminated_this_week", "judge_rank", "judge_percent",
        "season", "week", "celebrity_name"
    ]
    
    # 检查特征是否存在
    missing_cols = [col for col in feature_cols + target_cols if col not in valid_df.columns]
    if missing_cols:
        print(f"警告：以下列缺失: {missing_cols}")
        # 为缺失列创建默认值
        for col in missing_cols:
            if col in feature_cols:
                valid_df[col] = 0
    
    # 合并特征与目标变量，处理布尔值为整数
    model_df = valid_df[feature_cols + target_cols].copy()
    
    # 转换布尔类型为整数
    bool_cols = ["all_star_season", "controversial_contestant", "eliminated_this_week"]
    for col in bool_cols:
        if col in model_df.columns:
            model_df[col] = model_df[col].astype(int)
    
    # 按赛季划分投票机制（排名法vs百分比法）
    rank_seasons = list(range(1, 3)) + list(range(28, 35))  # 1-2、28-34赛季
    model_df["voting_mechanism"] = np.where(
        model_df["season"].isin(rank_seasons), "rank_based", "percent_based"
    )
    
    print(f"✓ 特征提取完成")
    print(f"  - 特征维度: {len(feature_cols)}")
    print(f"  - 排名法样本: {(model_df['voting_mechanism'] == 'rank_based').sum()} 条")
    print(f"  - 百分比法样本: {(model_df['voting_mechanism'] == 'percent_based').sum()} 条")
    
    return model_df


# ============================================================
# 三、分机制建模：粉丝投票估计
# ============================================================

def rank_based_vote_estimation(model_df):
    """
    排名法建模：整数规划求解粉丝排名
    
    适用赛季：1-2、28-34
    目标：最小化评委排名与粉丝排名差异，满足淘汰约束
    
    Parameters:
    -----------
    model_df : DataFrame
        建模数据
    
    Returns:
    --------
    rank_vote_results : DataFrame
        粉丝排名与投票数估计结果
    """
    print("\n" + "=" * 60)
    print("步骤3.1: 排名法建模（整数规划）")
    print("=" * 60)
    
    rank_df = model_df[model_df["voting_mechanism"] == "rank_based"].copy()
    
    if len(rank_df) == 0:
        print("✗ 无排名法数据，跳过")
        return pd.DataFrame()
    
    results = []
    success_count = 0
    fail_count = 0
    
    # 按赛季-周分组求解（每周独立优化）
    grouped = rank_df.groupby(["season", "week"])
    print(f"共 {len(grouped)} 个赛季-周需要求解\n")
    
    for (season, week), group in tqdm(grouped, desc="排名法优化求解"):
        n_contestants = len(group)
        contestants = group["celebrity_name"].tolist()
        judge_ranks = group["judge_rank"].tolist()
        is_eliminated = group["eliminated_this_week"].tolist()
        eliminated_idx = [i for i, val in enumerate(is_eliminated) if val]
        
        # 1. 构建整数规划问题
        prob = LpProblem(f"Fan_Vote_Rank_S{season}_W{week}", LpMinimize)
        
        # 2. 变量：粉丝排名（1~n_contestants）
        fan_ranks = [
            LpVariable(f"fan_rank_{i}", lowBound=1, upBound=n_contestants, cat=LpInteger)
            for i in range(n_contestants)
        ]
        
        # 3. 目标函数：最小化评委排名与粉丝排名的加权差异
        # 使用辅助变量来处理绝对值
        abs_diff_vars = []
        for i in range(n_contestants):
            diff_pos = LpVariable(f"diff_pos_{i}", lowBound=0, cat='Continuous')
            diff_neg = LpVariable(f"diff_neg_{i}", lowBound=0, cat='Continuous')
            prob += fan_ranks[i] - judge_ranks[i] == diff_pos - diff_neg
            abs_diff_vars.append(diff_pos + diff_neg)
        
        prob += lpSum(abs_diff_vars)
        
        # 4. 约束1：所有排名唯一（无并列）
        for i in range(n_contestants):
            for j in range(i+1, n_contestants):
                # 使用二进制变量实现不等约束
                binary_var = LpVariable(f"neq_{i}_{j}", cat='Binary')
                M = n_contestants  # 大M法的M值
                prob += fan_ranks[i] - fan_ranks[j] >= 1 - M * binary_var
                prob += fan_ranks[j] - fan_ranks[i] >= 1 - M * (1 - binary_var)
        
        # 5. 约束2：淘汰选手总排名（评委+粉丝）≥所有非淘汰选手
        if eliminated_idx:
            for e_idx in eliminated_idx:
                for n_idx in [i for i in range(n_contestants) if i not in eliminated_idx]:
                    prob += (judge_ranks[e_idx] + fan_ranks[e_idx]) >= \
                            (judge_ranks[n_idx] + fan_ranks[n_idx])
        
        # 6. 求解
        try:
            prob.solve()
            
            if LpStatus[prob.status] == 'Optimal':
                # 7. 提取结果
                fan_rank_results = [int(value(fan_ranks[i])) for i in range(n_contestants)]
                
                # 票数映射：V = k / rank（排名越高，票数越多）
                total_votes = 10000
                fan_votes = [total_votes / rank for rank in fan_rank_results]
                
                # 8. 存储结果
                week_results = pd.DataFrame({
                    "season": season,
                    "week": week,
                    "celebrity_name": contestants,
                    "fan_rank": fan_rank_results,
                    "fan_votes": fan_votes,
                    "voting_mechanism": "rank_based"
                })
                results.append(week_results)
                success_count += 1
            else:
                print(f"警告: S{season}W{week} 求解失败 - 状态: {LpStatus[prob.status]}")
                fail_count += 1
        
        except Exception as e:
            print(f"警告: S{season}W{week} 求解异常 - {e}")
            fail_count += 1
    
    print(f"\n✓ 排名法建模完成")
    print(f"  - 成功求解: {success_count}/{len(grouped)}")
    print(f"  - 失败案例: {fail_count}")
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


def percent_based_vote_estimation(model_df):
    """
    百分比法建模：贝叶斯模型 + MCMC采样
    
    适用赛季：3-27
    目标：求解粉丝百分比分布，含不确定性估计
    
    Parameters:
    -----------
    model_df : DataFrame
        建模数据
    
    Returns:
    --------
    percent_vote_results : DataFrame
        粉丝百分比后验（均值+置信区间）与投票数
    """
    print("\n" + "=" * 60)
    print("步骤3.2: 百分比法建模（贝叶斯-MCMC）")
    print("=" * 60)
    
    percent_df = model_df[model_df["voting_mechanism"] == "percent_based"].copy()
    
    if len(percent_df) == 0:
        print("✗ 无百分比法数据，跳过")
        return pd.DataFrame()
    
    results = []
    success_count = 0
    fail_count = 0
    
    grouped = percent_df.groupby(["season", "week"])
    print(f"共 {len(grouped)} 个赛季-周需要建模\n")
    
    for (season, week), group in tqdm(grouped, desc="百分比法贝叶斯建模"):
        n_contestants = len(group)
        contestants = group["celebrity_name"].tolist()
        judge_percent = group["judge_percent"].values
        is_eliminated = group["eliminated_this_week"].values
        eliminated_idx = np.where(is_eliminated)[0]
        non_eliminated_idx = np.where(~is_eliminated)[0]
        
        try:
            # 1. 构建贝叶斯模型
            with pm.Model() as model:
                # 先验：Dirichlet分布（基于人气特征调整）
                popularity_score = group["relative_level"].values
                alpha = popularity_score * 5 + 1  # 浓度参数
                fan_percent = pm.Dirichlet("fan_percent", a=alpha)
                
                # 约束：淘汰选手的总百分比≤非淘汰选手
                if len(eliminated_idx) > 0 and len(non_eliminated_idx) > 0:
                    for e_idx in eliminated_idx:
                        for n_idx in non_eliminated_idx[:min(3, len(non_eliminated_idx))]:  # 限制约束数量
                            total_e = judge_percent[e_idx] + fan_percent[e_idx]
                            total_n = judge_percent[n_idx] + fan_percent[n_idx]
                            pm.Potential(
                                f"constraint_{e_idx}_{n_idx}",
                                pm.math.switch(
                                    total_e <= total_n + 0.01,  # 容差
                                    0,
                                    -1000  # 惩罚项
                                )
                            )
                
                # 采样：NUTS算法
                trace = pm.sample(
                    1000, 
                    tune=500, 
                    cores=1, 
                    target_accept=0.9,
                    return_inferencedata=False,
                    progressbar=False
                )
            
            # 2. 后验分析
            fan_percent_samples = trace["fan_percent"]
            fan_percent_mean = fan_percent_samples.mean(axis=0)
            fan_percent_lower = np.percentile(fan_percent_samples, 2.5, axis=0)
            fan_percent_upper = np.percentile(fan_percent_samples, 97.5, axis=0)
            fan_percent_std = fan_percent_samples.std(axis=0)
            
            # 3. 映射为投票数
            total_votes = 10000
            fan_votes_mean = fan_percent_mean * total_votes
            fan_votes_lower = fan_percent_lower * total_votes
            fan_votes_upper = fan_percent_upper * total_votes
            vote_std = fan_percent_std * total_votes
            
            # 4. 存储结果
            week_results = pd.DataFrame({
                "season": season,
                "week": week,
                "celebrity_name": contestants,
                "fan_percent_mean": fan_percent_mean,
                "fan_percent_95ci_lower": fan_percent_lower,
                "fan_percent_95ci_upper": fan_percent_upper,
                "fan_votes_mean": fan_votes_mean,
                "fan_votes_95ci_lower": fan_votes_lower,
                "fan_votes_95ci_upper": fan_votes_upper,
                "voting_mechanism": "percent_based",
                "vote_std": vote_std
            })
            results.append(week_results)
            success_count += 1
        
        except Exception as e:
            print(f"警告: S{season}W{week} 建模异常 - {e}")
            fail_count += 1
    
    print(f"\n✓ 百分比法建模完成")
    print(f"  - 成功建模: {success_count}/{len(grouped)}")
    print(f"  - 失败案例: {fail_count}")
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


def merge_vote_results(model_df, rank_vote_results, percent_vote_results):
    """
    合并两种机制的投票结果
    """
    print("\n" + "=" * 60)
    print("步骤3.3: 合并投票结果")
    print("=" * 60)
    
    # 合并结果
    all_results = []
    if not rank_vote_results.empty:
        all_results.append(rank_vote_results)
    if not percent_vote_results.empty:
        all_results.append(percent_vote_results)
    
    if not all_results:
        print("✗ 无有效投票结果")
        return model_df
    
    all_vote_results = pd.concat(all_results, ignore_index=True)
    
    # 关联回原始模型数据
    final_vote_df = pd.merge(
        model_df[["season", "week", "celebrity_name", "eliminated_this_week", 
                  "judge_rank", "judge_percent", "relative_level", 
                  "celebrity_industry_encoded", "ballroom_partner_encoded",
                  "celebrity_age_during_season_bin_encoded", "season_stage_encoded",
                  "all_star_season", "controversial_contestant", "voting_mechanism"]],
        all_vote_results,
        on=["season", "week", "celebrity_name", "voting_mechanism"],
        how="left"
    )
    
    print(f"✓ 合并完成，共 {len(final_vote_df)} 条记录")
    
    return final_vote_df


# ============================================================
# 四、动态因素：HMM模型
# ============================================================

def add_hmm_dynamic_features(final_vote_df, valid_df):
    """
    HMM模型：捕捉选手人气随时间的变化
    
    Parameters:
    -----------
    final_vote_df : DataFrame
        投票结果数据
    valid_df : DataFrame
        原始有效数据
    
    Returns:
    --------
    final_vote_df_with_dynamic : DataFrame
        添加动态特征后的数据
    """
    print("\n" + "=" * 60)
    print("步骤4: HMM动态特征建模")
    print("=" * 60)
    
    # 准备HMM数据
    hmm_data = valid_df[["season", "celebrity_name", "week", "relative_level"]].copy()
    hmm_results = []
    success_count = 0
    
    # 按选手-赛季分组
    grouped = hmm_data.groupby(["season", "celebrity_name"])
    print(f"共 {len(grouped)} 个选手-赛季组合\n")
    
    for (season, celeb), group in tqdm(grouped, desc="HMM动态建模"):
        if len(group) < 3:  # 参赛周数过少，跳过
            continue
        
        try:
            # 排序并准备观测序列
            group = group.sort_values("week").reset_index(drop=True)
            obs = group["relative_level"].values.reshape(-1, 1)
            
            # 构建HMM（3个隐藏状态：低/中/高人气）
            model = hmm.GaussianHMM(
                n_components=3, 
                covariance_type="diag", 
                n_iter=100,
                random_state=42
            )
            model.fit(obs)
            
            # 预测隐藏状态
            hidden_states = model.predict(obs)
            
            # 存储结果
            week_results = pd.DataFrame({
                "season": season,
                "celebrity_name": celeb,
                "week": group["week"],
                "popularity_state": hidden_states
            })
            hmm_results.append(week_results)
            success_count += 1
        
        except Exception as e:
            # 静默跳过失败案例
            pass
    
    print(f"\n✓ HMM建模完成")
    print(f"  - 成功建模: {success_count}/{len(grouped)}")
    
    # 合并动态特征
    if hmm_results:
        hmm_df = pd.concat(hmm_results, ignore_index=True)
        final_vote_df_with_dynamic = pd.merge(
            final_vote_df, hmm_df,
            on=["season", "celebrity_name", "week"],
            how="left"
        )
        # 填充缺失值
        final_vote_df_with_dynamic["popularity_state"] = \
            final_vote_df_with_dynamic["popularity_state"].fillna(1)
    else:
        final_vote_df_with_dynamic = final_vote_df.copy()
        final_vote_df_with_dynamic["popularity_state"] = 1
    
    return final_vote_df_with_dynamic


# ============================================================
# 五、分层建模：随机森林分析
# ============================================================

def random_forest_stratified_model(final_vote_df):
    """
    随机森林：预测粉丝投票并分析特征重要性
    
    Parameters:
    -----------
    final_vote_df : DataFrame
        包含动态特征的投票数据
    
    Returns:
    --------
    rf_reg : RandomForestRegressor
        回归模型
    rf_clf : RandomForestClassifier
        分类模型
    feature_importance : DataFrame
        特征重要性分析结果
    """
    print("\n" + "=" * 60)
    print("步骤5: 随机森林分层建模")
    print("=" * 60)
    
    # 特征列
    rf_feature_cols = [
        "relative_level", "celebrity_industry_encoded", "ballroom_partner_encoded",
        "celebrity_age_during_season_bin_encoded", "season_stage_encoded",
        "all_star_season", "controversial_contestant", "popularity_state"
    ]
    
    # 准备数据（使用百分比法结果）
    rf_data = final_vote_df[
        (final_vote_df["voting_mechanism"] == "percent_based") &
        final_vote_df["fan_votes_mean"].notna()
    ].copy()
    
    # 同时准备排名法数据作为补充
    rf_data_rank = final_vote_df[
        (final_vote_df["voting_mechanism"] == "rank_based") &
        final_vote_df["fan_votes"].notna()
    ].copy()
    
    if not rf_data_rank.empty:
        rf_data_rank["fan_votes_mean"] = rf_data_rank["fan_votes"]
        rf_data = pd.concat([rf_data, rf_data_rank], ignore_index=True)
    
    if rf_data.empty or len(rf_data) < 10:
        print("✗ 数据不足，跳过随机森林建模")
        return None, None, None
    
    # 删除缺失值
    rf_data = rf_data.dropna(subset=rf_feature_cols + ["fan_votes_mean", "eliminated_this_week"])
    
    print(f"训练样本数: {len(rf_data)}")
    
    # 1. 回归模型：预测粉丝投票数
    X_reg = rf_data[rf_feature_cols]
    y_reg = rf_data["fan_votes_mean"]
    
    rf_reg = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf_reg.fit(X_reg, y_reg)
    
    y_pred_reg = rf_reg.predict(X_reg)
    rmse = np.sqrt(mean_squared_error(y_reg, y_pred_reg))
    r2 = rf_reg.score(X_reg, y_reg)
    
    print(f"  - 回归模型 RMSE: {rmse:.2f}")
    print(f"  - 回归模型 R²: {r2:.4f}")
    
    # 2. 分类模型：预测是否淘汰
    X_clf = rf_data[rf_feature_cols]
    y_clf = rf_data["eliminated_this_week"].astype(int)
    
    # 检查类别分布
    print(f"\n  淘汰样本分布:")
    print(f"    - 未淘汰: {(y_clf == 0).sum()} ({(y_clf == 0).sum()/len(y_clf)*100:.1f}%)")
    print(f"    - 已淘汰: {(y_clf == 1).sum()} ({(y_clf == 1).sum()/len(y_clf)*100:.1f}%)")
    
    # 使用 class_weight='balanced' 处理类别不平衡
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,  # 降低叶节点最小样本数
        class_weight='balanced',  # 自动平衡类别权重
        random_state=42
    )
    rf_clf.fit(X_clf, y_clf)
    
    y_pred_clf = rf_clf.predict(X_clf)
    acc = accuracy_score(y_clf, y_pred_clf)
    
    # 计算更多评估指标（针对不平衡数据）
    print(f"  - 分类模型准确率: {acc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_clf, y_pred_clf)
    print(f"\n  混淆矩阵:")
    print(f"    [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"     [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    # 召回率和精确率
    if (y_clf == 1).sum() > 0:
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  - 淘汰预测召回率: {recall:.4f}")
        print(f"  - 淘汰预测精确率: {precision:.4f}")
        print(f"  - 淘汰预测F1分数: {f1:.4f}")
    
    # 3. 特征重要性
    feature_importance = pd.DataFrame({
        "feature": rf_feature_cols,
        "vote_pred_importance": rf_reg.feature_importances_,
        "elimination_pred_importance": rf_clf.feature_importances_
    }).sort_values("vote_pred_importance", ascending=False)
    
    print("\n特征重要性（投票预测）:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['vote_pred_importance']:.4f}")
    
    # 4. 可视化
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x="vote_pred_importance", y="feature", data=feature_importance)
    plt.title("Feature Importance: Fan Vote Prediction")
    plt.xlabel("Importance")
    
    plt.subplot(1, 2, 2)
    sns.barplot(x="elimination_pred_importance", y="feature", data=feature_importance)
    plt.title("Feature Importance: Elimination Prediction")
    plt.xlabel("Importance")
    
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
    print("\n✓ 特征重要性图已保存: feature_importance.png")
    
    return rf_reg, rf_clf, feature_importance


# ============================================================
# 六、模型验证与一致性度量
# ============================================================

def model_validation(final_vote_df):
    """
    模型验证：淘汰结果一致性、预测准确率、不确定性评估
    
    Parameters:
    -----------
    final_vote_df : DataFrame
        最终投票结果数据
    
    Returns:
    --------
    validation_summary : dict
        验证指标汇总
    consistency_df : DataFrame
        一致性详细结果
    uncertainty_df : DataFrame
        不确定性分析结果
    """
    print("\n" + "=" * 60)
    print("步骤6: 模型验证与一致性度量")
    print("=" * 60)
    
    # 1. 淘汰结果一致性度量
    consistency_metrics = []
    
    for (season, week), group in final_vote_df.groupby(["season", "week"]):
        if group.empty:
            continue
        
        mechanism = group["voting_mechanism"].iloc[0]
        
        try:
            if mechanism == "rank_based":
                # 排名法：总排名最高者应被淘汰
                if "fan_rank" in group.columns and group["fan_rank"].notna().any():
                    group["total_rank"] = group["judge_rank"] + group["fan_rank"]
                    predicted_eliminated = group.loc[group["total_rank"].idxmax(), "celebrity_name"]
                else:
                    continue
            else:
                # 百分比法：总百分比最低者应被淘汰
                if "fan_percent_mean" in group.columns and group["fan_percent_mean"].notna().any():
                    group["total_percent"] = group["judge_percent"] + group["fan_percent_mean"]
                    predicted_eliminated = group.loc[group["total_percent"].idxmin(), "celebrity_name"]
                else:
                    continue
            
            # 实际淘汰者
            actual_eliminated = group[group["eliminated_this_week"] == 1]["celebrity_name"].tolist()
            
            if not actual_eliminated:
                continue
            
            # 一致性判断
            consistency = 1 if predicted_eliminated in actual_eliminated else 0
            
            consistency_metrics.append({
                "season": season,
                "week": week,
                "mechanism": mechanism,
                "consistency": consistency,
                "num_eliminated": len(actual_eliminated),
                "predicted": predicted_eliminated,
                "actual": actual_eliminated[0] if len(actual_eliminated) == 1 else str(actual_eliminated)
            })
        
        except Exception as e:
            # 静默跳过异常
            pass
    
    consistency_df = pd.DataFrame(consistency_metrics)
    
    if not consistency_df.empty:
        overall_consistency = consistency_df["consistency"].mean()
        rank_consistency = consistency_df[consistency_df["mechanism"] == "rank_based"]["consistency"].mean()
        percent_consistency = consistency_df[consistency_df["mechanism"] == "percent_based"]["consistency"].mean()
        
        print(f"  - 整体一致性: {overall_consistency:.2%}")
        print(f"  - 排名法一致性: {rank_consistency:.2%}")
        print(f"  - 百分比法一致性: {percent_consistency:.2%}")
    else:
        overall_consistency = 0
        rank_consistency = 0
        percent_consistency = 0
        print("  - 无法计算一致性（数据不足）")
    
    # 2. 不确定性度量（百分比法）
    uncertainty_df = final_vote_df[
        (final_vote_df["voting_mechanism"] == "percent_based") &
        final_vote_df["vote_std"].notna()
    ][["season", "week", "celebrity_name", "vote_std", "fan_votes_mean"]].copy()
    
    if not uncertainty_df.empty:
        uncertainty_df["cv"] = uncertainty_df["vote_std"] / (uncertainty_df["fan_votes_mean"] + 1e-6)
        avg_cv = uncertainty_df["cv"].mean()
        print(f"  - 平均相对不确定性 (CV): {avg_cv:.4f}")
        
        # 可视化不确定性分布
        plt.figure(figsize=(10, 6))
        sns.histplot(uncertainty_df["cv"], kde=True, bins=30)
        plt.title("Relative Uncertainty Distribution (Coefficient of Variation)")
        plt.xlabel("CV (std/mean)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig("uncertainty_distribution.png", dpi=300, bbox_inches='tight')
        print("  ✓ 不确定性分布图已保存: uncertainty_distribution.png")
    else:
        avg_cv = 0
        print("  - 无不确定性数据")
    
    # 汇总
    validation_summary = {
        "overall_consistency": overall_consistency,
        "rank_based_consistency": rank_consistency,
        "percent_based_consistency": percent_consistency,
        "avg_uncertainty_cv": avg_cv,
        "total_weeks_validated": len(consistency_df)
    }
    
    return validation_summary, consistency_df, uncertainty_df


# ============================================================
# 七、灵敏度分析
# ============================================================

def sensitivity_analysis(final_vote_df):
    """
    灵敏度分析：改变核心假设，观察投票估计变化
    
    Parameters:
    -----------
    final_vote_df : DataFrame
        最终投票结果
    
    Returns:
    --------
    sensitivity_df : DataFrame
        不同场景下的投票结果
    sensitivity_cv : DataFrame
        灵敏度变异系数
    """
    print("\n" + "=" * 60)
    print("步骤7: 灵敏度分析")
    print("=" * 60)
    
    # 改变总票数假设
    total_vote_scenarios = [5000, 10000, 15000]
    sensitivity_results = []
    
    for scenario in total_vote_scenarios:
        scenario_df = final_vote_df.copy()
        
        # 调整排名法投票数
        mask_rank = scenario_df["voting_mechanism"] == "rank_based"
        if "fan_votes" in scenario_df.columns:
            scenario_df.loc[mask_rank, "fan_votes_scaled"] = \
                scenario_df.loc[mask_rank, "fan_votes"] * (scenario / 10000)
        
        # 调整百分比法投票数
        mask_percent = scenario_df["voting_mechanism"] == "percent_based"
        if "fan_votes_mean" in scenario_df.columns:
            scenario_df.loc[mask_percent, "fan_votes_scaled"] = \
                scenario_df.loc[mask_percent, "fan_votes_mean"] * (scenario / 10000)
        
        scenario_df["total_vote_scenario"] = scenario
        sensitivity_results.append(scenario_df)
    
    sensitivity_df = pd.concat(sensitivity_results, ignore_index=True)
    
    # 计算变异系数
    if "fan_votes_scaled" in sensitivity_df.columns:
        sensitivity_cv = sensitivity_df.groupby(
            ["season", "week", "celebrity_name"]
        )["fan_votes_scaled"].agg(
            mean="mean", 
            std="std", 
            cv=lambda x: x.std() / (x.mean() + 1e-6)
        ).reset_index()
        
        avg_sensitivity = sensitivity_cv["cv"].mean()
        print(f"  - 平均灵敏度 (CV): {avg_sensitivity:.4f}")
        
        # 可视化
        plt.figure(figsize=(10, 6))
        percent_data = sensitivity_df[sensitivity_df["voting_mechanism"] == "percent_based"]
        if not percent_data.empty and "fan_votes_scaled" in percent_data.columns:
            sns.boxplot(
                x="total_vote_scenario", 
                y="fan_votes_scaled", 
                data=percent_data
            )
            plt.title("Fan Vote Sensitivity to Total Vote Assumption")
            plt.xlabel("Total Votes Scenario")
            plt.ylabel("Fan Votes")
            plt.tight_layout()
            plt.savefig("sensitivity_analysis.png", dpi=300, bbox_inches='tight')
            print("  ✓ 灵敏度分析图已保存: sensitivity_analysis.png")
    else:
        sensitivity_cv = pd.DataFrame()
        print("  - 无法进行灵敏度分析（数据不足）")
    
    return sensitivity_df, sensitivity_cv


# ============================================================
# 八、最终结果输出与可视化
# ============================================================

def save_and_visualize_results(final_vote_df, consistency_df, uncertainty_df, 
                                sensitivity_cv, feature_importance):
    """
    保存所有结果并生成可视化
    
    Parameters:
    -----------
    final_vote_df : DataFrame
        最终投票结果
    consistency_df : DataFrame
        一致性结果
    uncertainty_df : DataFrame
        不确定性结果
    sensitivity_cv : DataFrame
        灵敏度结果
    feature_importance : DataFrame
        特征重要性
    """
    print("\n" + "=" * 60)
    print("步骤8: 结果保存与可视化")
    print("=" * 60)
    
    # 1. 保存CSV文件
    final_vote_df.to_csv("fan_vote_estimates.csv", index=False, encoding='utf-8-sig')
    print("✓ 已保存: fan_vote_estimates.csv")
    
    if not consistency_df.empty:
        consistency_df.to_csv("elimination_consistency.csv", index=False, encoding='utf-8-sig')
        print("✓ 已保存: elimination_consistency.csv")
    
    if not uncertainty_df.empty:
        uncertainty_df.to_csv("vote_uncertainty.csv", index=False, encoding='utf-8-sig')
        print("✓ 已保存: vote_uncertainty.csv")
    
    if not sensitivity_cv.empty:
        sensitivity_cv.to_csv("sensitivity_cv.csv", index=False, encoding='utf-8-sig')
        print("✓ 已保存: sensitivity_cv.csv")
    
    if feature_importance is not None:
        feature_importance.to_csv("feature_importance.csv", index=False, encoding='utf-8-sig')
        print("✓ 已保存: feature_importance.csv")
    
    # 2. 生成综合可视化
    # 示例：百分比法后验分布（选择一个示例周）
    percent_sample = final_vote_df[
        (final_vote_df["voting_mechanism"] == "percent_based") &
        final_vote_df["fan_votes_mean"].notna()
    ]
    
    if not percent_sample.empty:
        # 选择第一个有效的赛季-周
        sample_group = percent_sample.groupby(["season", "week"]).first().reset_index().iloc[0]
        season_ex = sample_group["season"]
        week_ex = sample_group["week"]
        
        sample_data = percent_sample[
            (percent_sample["season"] == season_ex) &
            (percent_sample["week"] == week_ex)
        ]
        
        if len(sample_data) > 0 and "fan_votes_95ci_lower" in sample_data.columns:
            plt.figure(figsize=(12, 6))
            
            x_pos = range(len(sample_data))
            plt.errorbar(
                x_pos,
                sample_data["fan_votes_mean"],
                yerr=[
                    sample_data["fan_votes_mean"] - sample_data["fan_votes_95ci_lower"],
                    sample_data["fan_votes_95ci_upper"] - sample_data["fan_votes_mean"]
                ],
                fmt="o",
                capsize=5,
                capthick=2,
                markersize=8
            )
            
            plt.xticks(x_pos, sample_data["celebrity_name"], rotation=45, ha='right')
            plt.ylabel("Fan Votes")
            plt.title(f"Fan Vote Estimates with 95% CI\n(Season {int(season_ex)}, Week {int(week_ex)})")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig("fan_vote_ci_example.png", dpi=300, bbox_inches='tight')
            print("✓ 已保存: fan_vote_ci_example.png")
    
    print("\n" + "=" * 60)
    print("所有分析完成！")
    print("=" * 60)


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主执行函数
    """
    print("\n")
    print("=" * 60)
    print(" 舞蹈比赛粉丝投票分析系统 ".center(60, "="))
    print("=" * 60)
    print("\n")
    
    # 设置数据目录（根据实际情况修改）
    data_dir = "./MCM_C/Q1_final"
    
    # 如果目录不存在，尝试当前目录
    import os
    if not os.path.exists(data_dir):
        data_dir = "."
        print(f"注意：使用当前目录作为数据目录")
    
    try:
        # 二、数据加载与预处理
        final_df, valid_df, constraints_df, features_df = load_data(data_dir)
        model_df = extract_features(valid_df)
        
        # 三、分机制建模
        rank_vote_results = rank_based_vote_estimation(model_df)
        percent_vote_results = percent_based_vote_estimation(model_df)
        final_vote_df = merge_vote_results(model_df, rank_vote_results, percent_vote_results)
        
        # 四、动态因素建模
        final_vote_df = add_hmm_dynamic_features(final_vote_df, valid_df)
        
        # 五、随机森林分析
        rf_reg, rf_clf, feature_importance = random_forest_stratified_model(final_vote_df)
        
        # 六、模型验证
        validation_summary, consistency_df, uncertainty_df = model_validation(final_vote_df)
        
        # 七、灵敏度分析
        sensitivity_df, sensitivity_cv = sensitivity_analysis(final_vote_df)
        
        # 八、结果保存与可视化
        save_and_visualize_results(
            final_vote_df, consistency_df, uncertainty_df, 
            sensitivity_cv, feature_importance
        )
        
        # 打印最终汇总
        print("\n" + "=" * 60)
        print("验证指标汇总")
        print("=" * 60)
        for key, value in validation_summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        print("\n✓ 所有分析流程执行完毕！")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n✗ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
