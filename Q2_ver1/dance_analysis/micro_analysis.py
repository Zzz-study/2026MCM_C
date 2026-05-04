"""
争议选手微观分析模块
分析特定争议选手在不同机制下的表现差异
核心指标：
1. 排名相对差异 (ΔRank_rel)
2. 粉丝挽救效应系数 (γ)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def calculate_season_average_score(df_contestant, method='rank'):
    """
    计算选手的赛季平均得分
    
    Parameters:
    -----------
    df_contestant : DataFrame
        单个选手的完整赛季数据
    method : str
        'rank' 或 'percent'
    
    Returns:
    --------
    avg_score : float
        赛季平均得分
    weeks_data : DataFrame
        每周的得分数据
    """
    df_contestant = df_contestant.copy()
    weeks_data = []
    
    for _, row in df_contestant.iterrows():
        week = row['week']
        
        if method == 'rank':
            # 排名法：周度得分 = 评委排名 + 粉丝排名（越小越好）
            score = row['judge_rank'] + row['fan_rank']
        else:
            # 百分比法：周度得分 = 评委百分比 + 粉丝百分比（越大越好）
            score = row['judge_percent'] + row['fan_percent_mean']
        
        weeks_data.append({
            'week': week,
            'judge_rank': row['judge_rank'],
            'fan_rank': row.get('fan_rank', np.nan),
            'judge_percent': row['judge_percent'],
            'fan_percent': row.get('fan_percent_mean', np.nan),
            'score': score,
            'eliminated': row.get('eliminated_this_week', 0)
        })
    
    weeks_df = pd.DataFrame(weeks_data)
    avg_score = weeks_df['score'].mean()
    
    return avg_score, weeks_df


def calculate_final_ranking(season_df, method='rank', with_rule=False):
    """
    计算赛季最终排名（无规则情况下）
    
    Parameters:
    -----------
    season_df : DataFrame
        赛季数据
    method : str
        'rank' 或 'percent'
    with_rule : bool
        是否包含评委选择规则（暂不实现）
    
    Returns:
    --------
    ranking_df : DataFrame
        最终排名表
    """
    # 计算每个选手的赛季平均得分
    contestants = season_df['celebrity_name'].unique()
    ranking_list = []
    
    for contestant in contestants:
        contestant_df = season_df[season_df['celebrity_name'] == contestant]
        avg_score, _ = calculate_season_average_score(contestant_df, method)
        
        ranking_list.append({
            'celebrity_name': contestant,
            'avg_score': avg_score,
            'weeks_participated': len(contestant_df)
        })
    
    ranking_df = pd.DataFrame(ranking_list)
    
    # 按平均得分排序
    if method == 'rank':
        # 排名法：得分越小排名越高
        ranking_df = ranking_df.sort_values('avg_score', ascending=True)
    else:
        # 百分比法：得分越大排名越高
        ranking_df = ranking_df.sort_values('avg_score', ascending=False)
    
    # 添加排名
    ranking_df['final_rank'] = range(1, len(ranking_df) + 1)
    
    return ranking_df


def calculate_judge_only_ranking(season_df):
    """
    计算纯评委排名（不考虑粉丝投票）
    
    Parameters:
    -----------
    season_df : DataFrame
        赛季数据
    
    Returns:
    --------
    ranking_df : DataFrame
        纯评委排名表
    """
    contestants = season_df['celebrity_name'].unique()
    ranking_list = []
    
    for contestant in contestants:
        contestant_df = season_df[season_df['celebrity_name'] == contestant]
        # 使用评委排名的平均值
        avg_judge_rank = contestant_df['judge_rank'].mean()
        
        ranking_list.append({
            'celebrity_name': contestant,
            'avg_judge_rank': avg_judge_rank
        })
    
    ranking_df = pd.DataFrame(ranking_list)
    
    # 按评委平均排名排序（越小越好）
    ranking_df = ranking_df.sort_values('avg_judge_rank', ascending=True)
    ranking_df['judge_only_rank'] = range(1, len(ranking_df) + 1)
    
    return ranking_df


def analyze_controversial_case(df, player_name, season):
    """
    分析单个争议选手
    
    Parameters:
    -----------
    df : DataFrame
        完整数据
    player_name : str
        选手姓名
    season : int
        赛季编号
    
    Returns:
    --------
    metrics : dict
        {ΔRank_rel, γ_R, γ_P, 排名_R, 排名_P, Rank_Jonly}
    """
    # 获取该赛季数据
    season_df = df[df['season'] == season].copy()
    
    if len(season_df) == 0:
        return None
    
    # 获取该选手数据
    player_df = season_df[season_df['celebrity_name'] == player_name].copy()
    
    if len(player_df) == 0:
        print(f"  警告：未找到选手 {player_name} 在赛季 {season} 的数据")
        return None
    
    # 初始选手总数
    n_initial = len(season_df['celebrity_name'].unique())
    
    # 1. 计算两种方法的最终排名（无规则）
    ranking_rank = calculate_final_ranking(season_df, method='rank', with_rule=False)
    ranking_percent = calculate_final_ranking(season_df, method='percent', with_rule=False)
    
    # 2. 计算纯评委排名
    ranking_judge = calculate_judge_only_ranking(season_df)
    
    # 获取该选手的排名
    rank_R = ranking_rank[ranking_rank['celebrity_name'] == player_name]['final_rank'].values[0]
    rank_P = ranking_percent[ranking_percent['celebrity_name'] == player_name]['final_rank'].values[0]
    rank_J = ranking_judge[ranking_judge['celebrity_name'] == player_name]['judge_only_rank'].values[0]
    
    # 3. 计算排名相对差异 (ΔRank_rel)
    delta_rank_rel = abs(rank_R - rank_P) / n_initial * 100
    
    # 4. 计算粉丝挽救效应系数 (γ)
    # γ = (Rank_Jonly - Rank_method) / Rank_Jonly × 100%
    # 正值表示粉丝投票改善了排名，负值表示恶化了排名
    if rank_J > 0:
        gamma_R = (rank_J - rank_R) / rank_J * 100
        gamma_P = (rank_J - rank_P) / rank_J * 100
    else:
        gamma_R = 0
        gamma_P = 0
    
    metrics = {
        'player_name': player_name,
        'season': season,
        'n_initial': n_initial,
        'weeks_participated': len(player_df),
        'rank_R': rank_R,
        'rank_P': rank_P,
        'rank_J': rank_J,
        'delta_rank_rel': delta_rank_rel,
        'gamma_R': gamma_R,
        'gamma_P': gamma_P
    }
    
    return metrics


def analyze_all_controversial_cases(df):
    """
    分析所有争议选手
    
    Parameters:
    -----------
    df : DataFrame
        完整数据
    
    Returns:
    --------
    results_df : DataFrame
        争议选手分析结果
    """
    print("=" * 60)
    print("争议选手微观分析")
    print("=" * 60)
    
    # 争议选手列表
    controversial_cases = {
        "Jerry Rice": 2,
        "Billy Ray Cyrus": 4,
        "Bristol Palin": 11,
        "Bobby Bones": 27
    }
    
    results_list = []
    
    for player_name, season in controversial_cases.items():
        print(f"\n分析选手: {player_name} (赛季 {season})")
        
        metrics = analyze_controversial_case(df, player_name, season)
        
        if metrics:
            results_list.append(metrics)
            
            print(f"  初始选手数: {metrics['n_initial']}")
            print(f"  参赛周数: {metrics['weeks_participated']}")
            print(f"  排名法排名: {metrics['rank_R']}")
            print(f"  百分比法排名: {metrics['rank_P']}")
            print(f"  纯评委排名: {metrics['rank_J']}")
            print(f"  排名相对差异 (ΔRank_rel): {metrics['delta_rank_rel']:.2f}%")
            print(f"  粉丝挽救效应 (γ_R): {metrics['gamma_R']:.2f}%")
            print(f"  粉丝挽救效应 (γ_P): {metrics['gamma_P']:.2f}%")
            
            # 判断机制差异
            if metrics['delta_rank_rel'] < 20:
                print(f"  → 两种方法无本质差异")
            elif metrics['delta_rank_rel'] >= 40:
                if abs(metrics['gamma_R']) > abs(metrics['gamma_P']):
                    print(f"  → 排名法对该选手更有利")
                else:
                    print(f"  → 百分比法对该选手更有利")
            else:
                print(f"  → 两种方法有中等差异")
    
    results_df = pd.DataFrame(results_list)
    
    # 添加解释性判断
    if len(results_df) > 0:
        print("\n" + "=" * 60)
        print("综合分析")
        print("=" * 60)
        
        # 统计有显著差异的选手
        significant_diff = results_df[results_df['delta_rank_rel'] >= 40]
        if len(significant_diff) > 0:
            print(f"\n有显著机制差异的选手 (ΔRank_rel ≥ 40%):")
            for _, row in significant_diff.iterrows():
                print(f"  - {row['player_name']}: {row['delta_rank_rel']:.2f}%")
        
        # 比较粉丝挽救效应
        print(f"\n粉丝挽救效应对比:")
        print(f"  排名法平均 γ_R: {results_df['gamma_R'].mean():.2f}%")
        print(f"  百分比法平均 γ_P: {results_df['gamma_P'].mean():.2f}%")
        
        if results_df['gamma_R'].mean() > results_df['gamma_P'].mean():
            print(f"  → 排名法的粉丝挽救效应更强")
        else:
            print(f"  → 百分比法的粉丝挽救效应更强")
    
    return results_df


if __name__ == "__main__":
    # 测试代码
    from data_loader import load_fan_vote_estimates, preprocess_data
    
    df = load_fan_vote_estimates("../fan_vote_estimates.csv")
    df = preprocess_data(df)
    
    results_df = analyze_all_controversial_cases(df)
    
    # 保存结果
    results_df.to_csv("../输出/表格输出/micro_analysis.csv", index=False, encoding='utf-8-sig')
    
    print("\n[OK] 结果已保存")
