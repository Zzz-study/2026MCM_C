"""
28季特殊分析模块
28季首次使用"排名法+评委选择"规则，需要特殊分析
比较维度：
1. 纵向比较：28季 vs 前2季（排名法）
2. 横向比较：28季 vs 3-27季（百分比法）
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def compare_season28_vertical(df):
    """
    纵向比较：28季（排名法+规则） vs 前2季（排名法+无规则）
    
    Parameters:
    -----------
    df : DataFrame
        完整数据
    
    Returns:
    --------
    comparison_df : DataFrame
        纵向比较结果
    """
    from macro_comparison import calculate_fan_bias_metrics
    from rule_simulation import calculate_rule_impact
    
    print("=" * 60)
    print("28季纵向比较：排名法有规则 vs 无规则")
    print("=" * 60)
    
    # 前2季（排名法+无规则）
    early_seasons = [1, 2]
    early_metrics = []
    
    for season in early_seasons:
        metrics = calculate_fan_bias_metrics(df, season, method='rank')
        if metrics:
            early_metrics.append(metrics)
            print(f"\n赛季 {season} (排名法, 无规则):")
            print(f"  ρ={metrics['rho']:.4f}, W={metrics['W']:.4f}, D={metrics['D']:.4f}")
    
    # 28季（排名法+规则）
    print(f"\n赛季 28 (排名法, 有规则):")
    metrics_28 = calculate_fan_bias_metrics(df, 28, method='rank')
    
    if metrics_28:
        print(f"  ρ={metrics_28['rho']:.4f}, W={metrics_28['W']:.4f}, D={metrics_28['D']:.4f}")
        
        # 计算规则影响
        impact_28, _ = calculate_rule_impact(df, 28, method='rank')
        if impact_28:
            print(f"  平均排名变化: {impact_28['avg_rank_change']:.2f}")
            print(f"  高风险选手数: {impact_28['high_risk_players']}")
    
    # 计算差异
    if len(early_metrics) > 0 and metrics_28:
        avg_early_rho = np.mean([m['rho'] for m in early_metrics])
        avg_early_W = np.mean([m['W'] for m in early_metrics])
        avg_early_D = np.mean([m['D'] for m in early_metrics])
        
        comparison = {
            'metric': ['ρ (粉丝相关性)', 'W (粉丝权重)', 'D (粉丝偏差)'],
            'seasons_1_2_avg': [avg_early_rho, avg_early_W, avg_early_D],
            'season_28': [metrics_28['rho'], metrics_28['W'], metrics_28['D']],
            'difference': [
                metrics_28['rho'] - avg_early_rho,
                metrics_28['W'] - avg_early_W,
                metrics_28['D'] - avg_early_D
            ],
            'change_pct': [
                (metrics_28['rho'] - avg_early_rho) / (avg_early_rho + 1e-10) * 100,
                (metrics_28['W'] - avg_early_W) / (avg_early_W + 1e-10) * 100,
                (metrics_28['D'] - avg_early_D) / (avg_early_D + 1e-10) * 100
            ]
        }
        
        comparison_df = pd.DataFrame(comparison)
        
        print("\n" + "=" * 60)
        print("纵向对比总结")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        
        # 判断规则影响
        print("\n结论：")
        if abs(comparison_df['change_pct'].mean()) < 10:
            print("  规则对排名法的粉丝偏向性影响较小")
        elif comparison_df.loc[comparison_df['metric'].str.contains('ρ|W'), 'change_pct'].mean() < 0:
            print("  规则削弱了排名法的粉丝偏向性")
        else:
            print("  规则增强了排名法的粉丝偏向性")
        
        return comparison_df
    
    return None


def compare_season28_horizontal(df):
    """
    横向比较：28季（排名法+规则） vs 3-27季（百分比法+无规则）
    
    Parameters:
    -----------
    df : DataFrame
        完整数据
    
    Returns:
    --------
    comparison_df : DataFrame
        横向比较结果
    """
    from macro_comparison import calculate_fan_bias_metrics
    
    print("\n" + "=" * 60)
    print("28季横向比较：排名法+规则 vs 百分比法")
    print("=" * 60)
    
    # 3-27季（百分比法+无规则）
    percent_seasons = range(3, 28)
    percent_metrics = []
    
    for season in percent_seasons:
        metrics = calculate_fan_bias_metrics(df, season, method='percent')
        if metrics:
            percent_metrics.append(metrics)
    
    # 28季（排名法+规则）
    metrics_28 = calculate_fan_bias_metrics(df, 28, method='rank')
    
    if len(percent_metrics) > 0 and metrics_28:
        avg_percent_rho = np.mean([m['rho'] for m in percent_metrics])
        avg_percent_W = np.mean([m['W'] for m in percent_metrics])
        avg_percent_D = np.mean([m['D'] for m in percent_metrics])
        
        print(f"\n赛季 3-27 平均 (百分比法, 无规则):")
        print(f"  ρ={avg_percent_rho:.4f}, W={avg_percent_W:.4f}, D={avg_percent_D:.4f}")
        
        print(f"\n赛季 28 (排名法, 有规则):")
        print(f"  ρ={metrics_28['rho']:.4f}, W={metrics_28['W']:.4f}, D={metrics_28['D']:.4f}")
        
        # 计算标准化比值
        R_rho = metrics_28['rho'] / (avg_percent_rho + 1e-10) * 100
        R_W = metrics_28['W'] / (avg_percent_W + 1e-10) * 100
        R_D = (avg_percent_D + 1e-10) / (metrics_28['D'] + 1e-10) * 100  # D越小越好，所以反过来
        
        comparison = {
            'metric': ['ρ (粉丝相关性)', 'W (粉丝权重)', 'D (粉丝偏差)'],
            'seasons_3_27_avg': [avg_percent_rho, avg_percent_W, avg_percent_D],
            'season_28': [metrics_28['rho'], metrics_28['W'], metrics_28['D']],
            'ratio_pct': [R_rho, R_W, R_D],
            'interpretation': [
                '28季更高' if R_rho > 100 else '百分比法更高',
                '28季更高' if R_W > 100 else '百分比法更高',
                '28季更低(更好)' if R_D > 100 else '百分比法更低(更好)'
            ]
        }
        
        comparison_df = pd.DataFrame(comparison)
        
        print("\n" + "=" * 60)
        print("横向对比总结")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        
        # 判断
        print("\n结论：")
        favorable_to_28 = sum([
            R_rho > 100,  # ρ更高更好
            R_W > 100,    # W更高更好
            R_D > 100     # D更低更好（比值>100意味着28季的D更小）
        ])
        
        if favorable_to_28 >= 2:
            print(f"  28季（排名法+规则）在 {favorable_to_28}/3 个指标上优于百分比法")
            print("  → 规则下的排名法仍比百分比法更偏向粉丝")
        else:
            print(f"  28季（排名法+规则）在 {favorable_to_28}/3 个指标上优于百分比法")
            print("  → 规则下的排名法不如百分比法偏向粉丝")
        
        return comparison_df
    
    return None


def analyze_season28(df):
    """
    完整的28季分析
    
    Parameters:
    -----------
    df : DataFrame
        完整数据
    
    Returns:
    --------
    vertical_df : DataFrame
        纵向比较结果
    horizontal_df : DataFrame
        横向比较结果
    """
    print("=" * 60)
    print("28季特殊分析")
    print("=" * 60)
    print("28季首次使用'排名法+评委选择'规则")
    
    # 检查28季数据是否存在
    season_28_df = df[df['season'] == 28]
    
    if len(season_28_df) == 0:
        print("\n警告：未找到28季数据")
        return None, None
    
    print(f"\n28季数据: {len(season_28_df)} 条记录")
    print(f"选手数: {season_28_df['celebrity_name'].nunique()}")
    print(f"周数: {season_28_df['week'].nunique()}")
    
    # 纵向比较
    vertical_df = compare_season28_vertical(df)
    
    # 横向比较
    horizontal_df = compare_season28_horizontal(df)
    
    return vertical_df, horizontal_df


if __name__ == "__main__":
    # 测试代码
    from data_loader import load_fan_vote_estimates, preprocess_data
    
    df = load_fan_vote_estimates("../fan_vote_estimates.csv")
    df = preprocess_data(df)
    
    vertical_df, horizontal_df = analyze_season28(df)
    
    # 保存结果
    if vertical_df is not None:
        vertical_df.to_csv("../输出/表格输出/season28_vertical.csv", index=False, encoding='utf-8-sig')
    if horizontal_df is not None:
        horizontal_df.to_csv("../输出/表格输出/season28_horizontal.csv", index=False, encoding='utf-8-sig')
    
    print("\n[OK] 结果已保存")
