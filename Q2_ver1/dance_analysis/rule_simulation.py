"""
规则扩展分析模块 - 评委选择淘汰机制
模拟"评委每周从末两名选择淘汰"规则的影响
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def simulate_judge_choice_rule(season_df, method='rank'):
    """
    模拟评委选择淘汰规则
    
    规则说明：
    - 每周：计算所有选手的周度得分
    - 确定：周度得分最低的两位选手（末两名）
    - 淘汰：评委淘汰其中评委排名更差（数值更大）的选手
    - 迭代：剩余选手进入下一周，重复直至赛季结束
    
    Parameters:
    -----------
    season_df : DataFrame
        赛季完整数据
    method : str
        'rank' 或 'percent'
    
    Returns:
    --------
    final_ranking : DataFrame
        有规则的最终排名
    bottom2_count : dict
        每位选手进入末两名的次数
    elimination_order : list
        淘汰顺序
    """
    # 获取所有周数和选手
    weeks = sorted(season_df['week'].unique())
    all_contestants = set(season_df['celebrity_name'].unique())
    
    # 初始化
    current_players = all_contestants.copy()
    bottom2_count = {player: 0 for player in all_contestants}
    elimination_order = []  # (week, player, reason)
    
    for week in weeks:
        # 如果只剩2人或更少，停止淘汰
        if len(current_players) <= 2:
            break
        
        # 获取本周数据（只包含当前仍在比赛的选手）
        week_df = season_df[(season_df['week'] == week) & 
                            (season_df['celebrity_name'].isin(current_players))].copy()
        
        if len(week_df) < 2:
            continue
        
        # 计算本周得分
        if method == 'rank':
            # 排名法：得分 = 评委排名 + 粉丝排名（越小越好）
            week_df['week_score'] = week_df['judge_rank'] + week_df['fan_rank']
            # 按得分升序排序（得分高=排名差）
            week_df = week_df.sort_values('week_score', ascending=False)
        else:
            # 百分比法：得分 = 评委百分比 + 粉丝百分比（越大越好）
            week_df['week_score'] = week_df['judge_percent'] + week_df['fan_percent_mean']
            # 按得分降序排序（得分低=排名差）
            week_df = week_df.sort_values('week_score', ascending=True)
        
        # 确定末两名
        if len(week_df) >= 2:
            bottom_two = week_df.tail(2)
            
            # 记录进入末两名
            for player in bottom_two['celebrity_name']:
                bottom2_count[player] += 1
            
            # 评委淘汰选择：淘汰评委排名更差的选手（judge_rank更大）
            bottom_two_sorted = bottom_two.sort_values('judge_rank', ascending=False)
            eliminated_player = bottom_two_sorted.iloc[0]['celebrity_name']
            
            # 记录淘汰
            elimination_order.append({
                'week': week,
                'eliminated_player': eliminated_player,
                'judge_rank': bottom_two_sorted.iloc[0]['judge_rank'],
                'week_score': bottom_two_sorted.iloc[0]['week_score']
            })
            
            # 从当前选手池移除
            current_players.remove(eliminated_player)
    
    # 生成最终排名
    # 决赛选手（未被淘汰的）按最后一周得分排序
    final_week = weeks[-1]
    final_week_df = season_df[(season_df['week'] == final_week) & 
                               (season_df['celebrity_name'].isin(current_players))].copy()
    
    if len(final_week_df) > 0:
        if method == 'rank':
            final_week_df['week_score'] = final_week_df['judge_rank'] + final_week_df['fan_rank']
            final_week_df = final_week_df.sort_values('week_score', ascending=True)
        else:
            final_week_df['week_score'] = final_week_df['judge_percent'] + final_week_df['fan_percent_mean']
            final_week_df = final_week_df.sort_values('week_score', ascending=False)
        
        # 决赛选手排名
        final_ranking_list = []
        for rank, (_, row) in enumerate(final_week_df.iterrows(), 1):
            final_ranking_list.append({
                'celebrity_name': row['celebrity_name'],
                'final_rank': rank,
                'final_score': row['week_score'],
                'elimination_week': None
            })
        
        # 淘汰选手按淘汰倒序排名
        next_rank = len(final_ranking_list) + 1
        for elim_info in reversed(elimination_order):
            final_ranking_list.append({
                'celebrity_name': elim_info['eliminated_player'],
                'final_rank': next_rank,
                'final_score': elim_info['week_score'],
                'elimination_week': elim_info['week']
            })
            next_rank += 1
    else:
        # 如果没有决赛周数据，所有选手按淘汰倒序
        final_ranking_list = []
        for rank, elim_info in enumerate(reversed(elimination_order), 1):
            final_ranking_list.append({
                'celebrity_name': elim_info['eliminated_player'],
                'final_rank': rank,
                'final_score': elim_info['week_score'],
                'elimination_week': elim_info['week']
            })
    
    final_ranking = pd.DataFrame(final_ranking_list)
    
    return final_ranking, bottom2_count, elimination_order


def calculate_rule_impact(df, season, method='rank'):
    """
    计算规则对特定赛季的影响
    
    Parameters:
    -----------
    df : DataFrame
        完整数据
    season : int
        赛季编号
    method : str
        'rank' 或 'percent'
    
    Returns:
    --------
    impact_metrics : dict
        规则影响指标
    """
    from micro_analysis import calculate_final_ranking
    
    season_df = df[df['season'] == season].copy()
    
    if len(season_df) == 0:
        return None
    
    # 1. 计算无规则排名
    ranking_no_rule = calculate_final_ranking(season_df, method=method, with_rule=False)
    
    # 2. 计算有规则排名
    ranking_with_rule, bottom2_count, elimination_order = simulate_judge_choice_rule(season_df, method=method)
    
    # 3. 比较排名差异
    # 合并两个排名表
    ranking_comparison = ranking_no_rule[['celebrity_name', 'final_rank']].merge(
        ranking_with_rule[['celebrity_name', 'final_rank']],
        on='celebrity_name',
        suffixes=('_no_rule', '_with_rule')
    )
    
    # 计算排名变化
    ranking_comparison['rank_change'] = (ranking_comparison['final_rank_no_rule'] - 
                                          ranking_comparison['final_rank_with_rule'])
    
    # 4. 计算淘汰风险系数 (η)
    total_weeks = len(season_df['week'].unique())
    risk_metrics = []
    
    for player, count in bottom2_count.items():
        player_weeks = len(season_df[season_df['celebrity_name'] == player])
        if player_weeks > 0:
            eta = count / player_weeks * 100
            risk_metrics.append({
                'celebrity_name': player,
                'bottom2_count': count,
                'total_weeks': player_weeks,
                'eta': eta
            })
    
    risk_df = pd.DataFrame(risk_metrics)
    
    # 合并风险指标
    ranking_comparison = ranking_comparison.merge(risk_df, on='celebrity_name', how='left')
    
    impact_metrics = {
        'season': season,
        'method': method,
        'total_contestants': len(ranking_comparison),
        'avg_rank_change': ranking_comparison['rank_change'].abs().mean(),
        'max_rank_change': ranking_comparison['rank_change'].abs().max(),
        'avg_eta': ranking_comparison['eta'].mean(),
        'high_risk_players': len(ranking_comparison[ranking_comparison['eta'] >= 30])
    }
    
    return impact_metrics, ranking_comparison


def analyze_controversial_with_rule(df):
    """
    分析争议选手在有规则情况下的表现
    
    Parameters:
    -----------
    df : DataFrame
        完整数据
    
    Returns:
    --------
    results_df : DataFrame
        规则影响分析结果
    """
    from micro_analysis import analyze_controversial_case
    
    print("=" * 60)
    print("规则扩展分析 - 评委选择淘汰机制")
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
        
        season_df = df[df['season'] == season].copy()
        
        # 1. 无规则情况（已在micro_analysis中计算）
        metrics_no_rule = analyze_controversial_case(df, player_name, season)
        
        # 2. 有规则情况 - 排名法
        ranking_rank_rule, bottom2_rank, _ = simulate_judge_choice_rule(season_df, method='rank')
        rank_R_rule = ranking_rank_rule[ranking_rank_rule['celebrity_name'] == player_name]['final_rank'].values
        rank_R_rule = rank_R_rule[0] if len(rank_R_rule) > 0 else None
        
        # 3. 有规则情况 - 百分比法
        ranking_percent_rule, bottom2_percent, _ = simulate_judge_choice_rule(season_df, method='percent')
        rank_P_rule = ranking_percent_rule[ranking_percent_rule['celebrity_name'] == player_name]['final_rank'].values
        rank_P_rule = rank_P_rule[0] if len(rank_P_rule) > 0 else None
        
        if metrics_no_rule and rank_R_rule is not None and rank_P_rule is not None:
            # 计算规则影响
            n_initial = metrics_no_rule['n_initial']
            
            # 计算有规则的 ΔRank_rel
            delta_rank_rel_rule = abs(rank_R_rule - rank_P_rule) / n_initial * 100
            
            # 计算变化率
            delta_rank_rel_change = (delta_rank_rel_rule - metrics_no_rule['delta_rank_rel']) / (metrics_no_rule['delta_rank_rel'] + 1e-10) * 100
            
            # 淘汰风险
            player_weeks = len(season_df[season_df['celebrity_name'] == player_name])
            eta_R = bottom2_rank.get(player_name, 0) / player_weeks * 100 if player_weeks > 0 else 0
            eta_P = bottom2_percent.get(player_name, 0) / player_weeks * 100 if player_weeks > 0 else 0
            
            result = {
                'player_name': player_name,
                'season': season,
                # 无规则
                'rank_R_no_rule': metrics_no_rule['rank_R'],
                'rank_P_no_rule': metrics_no_rule['rank_P'],
                'delta_rank_rel_no_rule': metrics_no_rule['delta_rank_rel'],
                'gamma_R_no_rule': metrics_no_rule['gamma_R'],
                'gamma_P_no_rule': metrics_no_rule['gamma_P'],
                # 有规则
                'rank_R_with_rule': rank_R_rule,
                'rank_P_with_rule': rank_P_rule,
                'delta_rank_rel_with_rule': delta_rank_rel_rule,
                # 变化率
                'delta_rank_rel_change': delta_rank_rel_change,
                # 淘汰风险
                'eta_R': eta_R,
                'eta_P': eta_P
            }
            
            results_list.append(result)
            
            print(f"  无规则: 排名法排名={metrics_no_rule['rank_R']}, 百分比法排名={metrics_no_rule['rank_P']}")
            print(f"  有规则: 排名法排名={rank_R_rule}, 百分比法排名={rank_P_rule}")
            print(f"  ΔRank_rel变化: {metrics_no_rule['delta_rank_rel']:.2f}% → {delta_rank_rel_rule:.2f}% (变化率 {delta_rank_rel_change:.2f}%)")
            print(f"  淘汰风险 (η): 排名法={eta_R:.2f}%, 百分比法={eta_P:.2f}%")
    
    results_df = pd.DataFrame(results_list)
    
    if len(results_df) > 0:
        print("\n" + "=" * 60)
        print("综合分析")
        print("=" * 60)
        
        print(f"\n规则对ΔRank_rel的平均影响: {results_df['delta_rank_rel_change'].mean():.2f}%")
        print(f"平均淘汰风险 (η):")
        print(f"  排名法: {results_df['eta_R'].mean():.2f}%")
        print(f"  百分比法: {results_df['eta_P'].mean():.2f}%")
        
        # 判断规则影响
        if abs(results_df['delta_rank_rel_change'].mean()) < 10:
            print(f"\n→ 规则对方法差异影响较小")
        elif results_df['delta_rank_rel_change'].mean() > 0:
            print(f"\n→ 规则扩大了两种方法的差异")
        else:
            print(f"\n→ 规则缩小了两种方法的差异")
    
    return results_df


if __name__ == "__main__":
    # 测试代码
    from data_loader import load_fan_vote_estimates, preprocess_data
    
    df = load_fan_vote_estimates("../fan_vote_estimates.csv")
    df = preprocess_data(df)
    
    results_df = analyze_controversial_with_rule(df)
    
    # 保存结果
    results_df.to_csv("../输出/表格输出/rule_impact.csv", index=False, encoding='utf-8-sig')
    
    print("\n[OK] 结果已保存")
