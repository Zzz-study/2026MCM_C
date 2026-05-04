"""
舞蹈比赛粉丝投票分析系统 - 快速测试版本
仅处理前10个赛季-周组合，用于验证代码流程
"""

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 优化建模（排名法）
from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, lpSum, LpStatus, value

# 设置随机种子
np.random.seed(42)

print("=" * 60)
print("舞蹈比赛粉丝投票分析 - 快速测试版本")
print("=" * 60)

# 步骤1: 加载数据
print("\n步骤1: 加载数据...")
final_df = pd.read_csv("dance_competition_final_processed.csv", encoding='utf-8', encoding_errors='ignore')
print(f"  已加载 {len(final_df)} 行数据")

with open("fan_vote_constraints.json", "r", encoding='utf-8') as f:
    vote_constraints = json.load(f)
print(f"  已加载 {len(vote_constraints)} 条约束")

# 转换约束为DataFrame
constraints_list = []
for item in vote_constraints:
    contestant_names = item.get("contestant_list", [])
    eliminated_names = item.get("eliminated_list", [])
    judge_ranks = item.get("judge_ranks", {})
    judge_percents = item.get("judge_percents", {})
    
    df_item = pd.DataFrame({
        "season": item["season"],
        "week": item["week"],
        "celebrity_name": contestant_names,
        "is_eliminated": [name in eliminated_names for name in contestant_names],
        "judge_rank": [judge_ranks.get(name, 0) for name in contestant_names],
        "judge_percent": [judge_percents.get(name, 0.0) for name in contestant_names]
    })
    constraints_list.append(df_item)

constraints_df = pd.concat(constraints_list, ignore_index=True)
print(f"  约束数据转换完成: {len(constraints_df)} 行")

# 合并数据
merged_df = pd.merge(
    final_df, 
    constraints_df, 
    on=["season", "week", "celebrity_name"], 
    how="inner"
)

valid_df = merged_df[
    (merged_df["in_competition"] == True) & 
    (merged_df["non_competition_week"] == False) & 
    (merged_df["week"] <= merged_df["week_eliminated"])
].reset_index(drop=True)

print(f"  有效样本: {len(valid_df)} 行")

# 步骤2: 提取特征
print("\n步骤2: 提取特征...")
feature_cols = [
    "relative_level",
    "celebrity_industry_encoded",
    "ballroom_partner_encoded",
    "celebrity_age_during_season_bin_encoded",
    "season_stage_encoded",
    "all_star_season",
    "controversial_contestant"
]

target_cols = [
    "eliminated_this_week", "judge_rank", "judge_percent",
    "season", "week", "celebrity_name"
]

# 检查缺失列
missing_cols = [col for col in feature_cols if col not in valid_df.columns]
if missing_cols:
    print(f"  警告：缺失列 {missing_cols}，使用默认值0")
    for col in missing_cols:
        valid_df[col] = 0

model_df = valid_df[feature_cols + target_cols].copy()

# 转换布尔类型
bool_cols = ["all_star_season", "controversial_contestant", "eliminated_this_week"]
for col in bool_cols:
    if col in model_df.columns and model_df[col].dtype == 'bool':
        model_df[col] = model_df[col].astype(int)

# 划分投票机制
rank_seasons = list(range(1, 3)) + list(range(28, 35))
model_df["voting_mechanism"] = np.where(
    model_df["season"].isin(rank_seasons), "rank_based", "percent_based"
)

print(f"  特征提取完成")
print(f"  - 排名法样本: {(model_df['voting_mechanism'] == 'rank_based').sum()}")
print(f"  - 百分比法样本: {(model_df['voting_mechanism'] == 'percent_based').sum()}")

# 步骤3: 排名法建模（仅处理前5个组）
print("\n步骤3: 排名法建模（测试前5组）...")
rank_df = model_df[model_df["voting_mechanism"] == "rank_based"].copy()
results = []

if len(rank_df) > 0:
    grouped = list(rank_df.groupby(["season", "week"]))[:5]  # 仅前5组
    
    for (season, week), group in tqdm(grouped, desc="排名法优化"):
        n_contestants = len(group)
        contestants = group["celebrity_name"].tolist()
        judge_ranks = group["judge_rank"].tolist()
        is_eliminated = group["eliminated_this_week"].tolist()
        eliminated_idx = [i for i, val in enumerate(is_eliminated) if val]
        
        # 构建整数规划
        prob = LpProblem(f"Fan_Vote_S{season}_W{week}", LpMinimize)
        
        # 变量：粉丝排名
        fan_ranks = [
            LpVariable(f"fr_{i}", lowBound=1, upBound=n_contestants, cat=LpInteger)
            for i in range(n_contestants)
        ]
        
        # 目标函数：最小化差异（使用辅助变量处理绝对值）
        abs_diff_vars = []
        for i in range(n_contestants):
            diff_pos = LpVariable(f"dp_{i}", lowBound=0)
            diff_neg = LpVariable(f"dn_{i}", lowBound=0)
            prob += fan_ranks[i] - judge_ranks[i] == diff_pos - diff_neg
            abs_diff_vars.append(diff_pos + diff_neg)
        
        prob += lpSum(abs_diff_vars)
        
        # 约束1：排名唯一（使用二进制变量）
        for i in range(n_contestants):
            for j in range(i+1, n_contestants):
                binary_var = LpVariable(f"b_{i}_{j}", cat='Binary')
                M = n_contestants
                prob += fan_ranks[i] - fan_ranks[j] >= 1 - M * binary_var
                prob += fan_ranks[j] - fan_ranks[i] >= 1 - M * (1 - binary_var)
        
        # 约束2：淘汰约束
        if eliminated_idx:
            for e_idx in eliminated_idx:
                for n_idx in [i for i in range(n_contestants) if i not in eliminated_idx]:
                    prob += (judge_ranks[e_idx] + fan_ranks[e_idx]) >= \
                            (judge_ranks[n_idx] + fan_ranks[n_idx])
        
        # 求解
        try:
            prob.solve()
            
            if LpStatus[prob.status] == 'Optimal':
                fan_rank_results = [int(value(fan_ranks[i])) for i in range(n_contestants)]
                total_votes = 10000
                fan_votes = [total_votes / rank for rank in fan_rank_results]
                
                week_results = pd.DataFrame({
                    "season": season,
                    "week": week,
                    "celebrity_name": contestants,
                    "fan_rank": fan_rank_results,
                    "fan_votes": fan_votes,
                    "voting_mechanism": "rank_based"
                })
                results.append(week_results)
                print(f"  [OK] S{season}W{week} 求解成功")
            else:
                print(f"  [FAIL] S{season}W{week} 求解失败: {LpStatus[prob.status]}")
        
        except Exception as e:
            print(f"  [ERROR] S{season}W{week} 异常: {e}")
    
    if results:
        rank_vote_results = pd.concat(results, ignore_index=True)
        print(f"\n  排名法完成: {len(rank_vote_results)} 条记录")
        print(f"  示例结果:")
        print(rank_vote_results.head(3))
    else:
        print("\n  排名法无有效结果")
else:
    print("  无排名法数据")

# 步骤4: 保存结果
if results:
    rank_vote_results.to_csv("quick_test_results.csv", index=False, encoding='utf-8-sig')
    print("\n结果已保存到: quick_test_results.csv")

print("\n" + "=" * 60)
print("快速测试完成！")
print("=" * 60)
print("\n如果测试成功，可以运行完整版本:")
print("  python dance_competition_analysis.py")
