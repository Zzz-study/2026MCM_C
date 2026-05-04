"""
宏观比较分析模块 - 量化粉丝偏向性
计算三个核心指标：
1. 粉丝相关性系数 (ρ) - Spearman秩相关
2. 粉丝贡献权重 (W) - 回归反推
3. 纯粉丝偏差 (D) - Kendall tau距离
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


def calculate_combined_rankings(df_week, method='rank'):
    """
    计算两种方法的组合排名
    
    Parameters:
    -----------
    df_week : DataFrame
        单周数据
    method : str
        'rank' 或 'percent'
    
    Returns:
    --------
    df_week : DataFrame
        添加了组合得分和排名的数据
    """
    df_week = df_week.copy()
    
    if method == 'rank':
        # 排名法：Score = 评委排名 + 粉丝排名 (越小越好)
        df_week['combined_score'] = df_week['judge_rank'] + df_week['fan_rank']
        # 按得分升序排名（得分越小排名越高）
        df_week['combined_rank'] = df_week['combined_score'].rank(method='min', ascending=True)
    else:
        # 百分比法：Score = 评委百分比 + 粉丝百分比 (越大越好)
        df_week['combined_score'] = df_week['judge_percent'] + df_week['fan_percent_mean']
        # 按得分降序排名（得分越大排名越高）
        df_week['combined_rank'] = df_week['combined_score'].rank(method='min', ascending=False)
    
    return df_week


def calculate_pure_fan_ranking(df_week, method='rank'):
    """
    计算纯粉丝排名
    
    Parameters:
    -----------
    df_week : DataFrame
        单周数据
    method : str
        'rank' 或 'percent'
    
    Returns:
    --------
    df_week : DataFrame
        添加了纯粉丝排名的数据
    """
    df_week = df_week.copy()
    
    if method == 'rank':
        # 排名法：直接使用粉丝排名
        df_week['pure_fan_rank'] = df_week['fan_rank']
    else:
        # 百分比法：按粉丝百分比降序排名
        df_week['pure_fan_rank'] = df_week['fan_percent_mean'].rank(method='min', ascending=False)
    
    return df_week


def calculate_spearman_correlation(df_week, method='rank'):
    """
    计算粉丝相关性系数 (ρ) - Spearman秩相关
    
    Parameters:
    -----------
    df_week : DataFrame
        单周数据（需包含 combined_rank 和 pure_fan_rank）
    method : str
        'rank' 或 'percent'
    
    Returns:
    --------
    rho : float
        Spearman相关系数
    """
    # 使用纯粉丝排名和组合排名计算相关性
    # 注意：两者都是排名（越小越好），所以正相关意味着一致性高
    rho, _ = spearmanr(df_week['pure_fan_rank'], df_week['combined_rank'])
    
    return rho


def calculate_fan_weight_regression(df_week, method='rank'):
    """
    通过回归反推粉丝贡献权重 (W)
    
    模型：
    - 排名法：combined_rank = α * judge_rank + β * fan_rank + ε
    - 百分比法：combined_score = α * judge_percent + β * fan_percent + ε
    
    粉丝权重 W = β / (α + β)
    
    Parameters:
    -----------
    df_week : DataFrame
        单周数据
    method : str
        'rank' 或 'percent'
    
    Returns:
    --------
    W : float
        粉丝贡献权重 [0, 1]
    """
    df_week = df_week.copy()
    
    if method == 'rank':
        # 排名法：预测组合得分（评委排名 + 粉丝排名）
        X = df_week[['judge_rank', 'fan_rank']].values
        y = df_week['combined_score'].values
    else:
        # 百分比法：预测组合得分（评委百分比 + 粉丝百分比）
        X = df_week[['judge_percent', 'fan_percent_mean']].values
        y = df_week['combined_score'].values
    
    # 不使用截距，强制通过原点（因为0+0应该=0）
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    
    # 获取系数
    alpha = model.coef_[0]  # 评委权重系数
    beta = model.coef_[1]   # 粉丝权重系数
    
    # 计算粉丝贡献权重
    if alpha + beta > 0:
        W = beta / (alpha + beta)
    else:
        W = 0.5  # 如果权重都是负数，返回0.5作为默认值
    
    # 确保W在[0, 1]范围内
    W = np.clip(W, 0, 1)
    
    return W


def calculate_kendall_distance(df_week, method='rank'):
    """
    计算纯粉丝偏差 (D) - 基于Kendall tau距离
    
    D = 1 - Kendall(最终排名, 纯粉丝排名)
    D ∈ [0, 1]，数值越小越偏向粉丝
    
    Parameters:
    -----------
    df_week : DataFrame
        单周数据（需包含 combined_rank 和 pure_fan_rank）
    method : str
        'rank' 或 'percent'
    
    Returns:
    --------
    D : float
        纯粉丝偏差
    """
    tau, _ = kendalltau(df_week['combined_rank'], df_week['pure_fan_rank'])
    
    # Kendall tau ∈ [-1, 1]，1表示完全一致
    # D = 1 - tau，使得D=0表示完全一致（最偏向粉丝）
    D = 1 - tau
    
    # 确保D在[0, 1]范围内
    D = np.clip(D, 0, 1)
    
    return D


def calculate_fan_bias_metrics(df, season, method='rank'):
    """
    计算单个赛季的粉丝偏向性指标
    
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
    metrics : dict
        {rho, W, D, n_weeks, n_contestants}
    """
    # 筛选该赛季数据
    season_df = df[df['season'] == season].copy()
    
    if len(season_df) == 0:
        return None
    
    # 按周计算指标，然后取平均
    weeks = season_df['week'].unique()
    rho_list = []
    W_list = []
    D_list = []
    
    for week in weeks:
        week_df = season_df[season_df['week'] == week].copy()
        
        # 需要至少3个选手才能计算
        if len(week_df) < 3:
            continue
        
        # 检查必要的列是否有缺失值
        if method == 'rank':
            required_cols = ['judge_rank', 'fan_rank']
        else:
            required_cols = ['judge_percent', 'fan_percent_mean']
        
        if week_df[required_cols].isnull().any().any():
            continue
        
        # 计算组合排名
        week_df = calculate_combined_rankings(week_df, method)
        
        # 计算纯粉丝排名
        week_df = calculate_pure_fan_ranking(week_df, method)
        
        # 计算三个指标
        try:
            rho = calculate_spearman_correlation(week_df, method)
            W = calculate_fan_weight_regression(week_df, method)
            D = calculate_kendall_distance(week_df, method)
            
            rho_list.append(rho)
            W_list.append(W)
            D_list.append(D)
        except:
            continue
    
    if len(rho_list) == 0:
        return None
    
    # 返回平均值
    metrics = {
        'season': season,
        'method': method,
        'rho': np.mean(rho_list),
        'rho_std': np.std(rho_list),
        'W': np.mean(W_list),
        'W_std': np.std(W_list),
        'D': np.mean(D_list),
        'D_std': np.std(D_list),
        'n_weeks': len(rho_list),
        'n_contestants': len(season_df['celebrity_name'].unique())
    }
    
    return metrics


def compare_all_seasons(df):
    """
    对所有赛季进行宏观比较分析
    
    Parameters:
    -----------
    df : DataFrame
        完整数据
    
    Returns:
    --------
    results_df : DataFrame
        各赛季的指标结果
    summary_df : DataFrame
        跨赛季汇总统计
    """
    print("=" * 60)
    print("宏观比较分析 - 计算粉丝偏向性指标")
    print("=" * 60)
    
    seasons = sorted(df['season'].unique())
    results_list = []
    
    for season in seasons:
        print(f"\n处理赛季 {season}...")
        
        # 排名法
        metrics_rank = calculate_fan_bias_metrics(df, season, method='rank')
        if metrics_rank:
            results_list.append(metrics_rank)
            print(f"  排名法: ρ={metrics_rank['rho']:.4f}, W={metrics_rank['W']:.4f}, D={metrics_rank['D']:.4f}")
        
        # 百分比法
        metrics_percent = calculate_fan_bias_metrics(df, season, method='percent')
        if metrics_percent:
            results_list.append(metrics_percent)
            print(f"  百分比法: ρ={metrics_percent['rho']:.4f}, W={metrics_percent['W']:.4f}, D={metrics_percent['D']:.4f}")
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results_list)
    
    # 计算跨赛季汇总
    summary_list = []
    for method in ['rank', 'percent']:
        method_df = results_df[results_df['method'] == method]
        if len(method_df) > 0:
            summary = {
                'method': method,
                'rho_mean': method_df['rho'].mean(),
                'rho_std': method_df['rho'].std(),
                'W_mean': method_df['W'].mean(),
                'W_std': method_df['W'].std(),
                'D_mean': method_df['D'].mean(),
                'D_std': method_df['D'].std(),
                'n_seasons': len(method_df)
            }
            summary_list.append(summary)
    
    summary_df = pd.DataFrame(summary_list)
    
    print("\n" + "=" * 60)
    print("跨赛季汇总统计")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    # 判断哪种方法更偏向粉丝
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    
    if len(summary_df) == 2:
        rank_row = summary_df[summary_df['method'] == 'rank'].iloc[0]
        percent_row = summary_df[summary_df['method'] == 'percent'].iloc[0]
        
        # 比较三个指标
        # ρ: 越大越偏向粉丝
        # W: 越大越偏向粉丝
        # D: 越小越偏向粉丝
        
        rho_winner = "排名法" if rank_row['rho_mean'] > percent_row['rho_mean'] else "百分比法"
        W_winner = "排名法" if rank_row['W_mean'] > percent_row['W_mean'] else "百分比法"
        D_winner = "排名法" if rank_row['D_mean'] < percent_row['D_mean'] else "百分比法"
        
        print(f"粉丝相关性系数 (ρ): {rho_winner} 更优")
        print(f"  排名法: {rank_row['rho_mean']:.4f} ± {rank_row['rho_std']:.4f}")
        print(f"  百分比法: {percent_row['rho_mean']:.4f} ± {percent_row['rho_std']:.4f}")
        
        print(f"\n粉丝贡献权重 (W): {W_winner} 更优")
        print(f"  排名法: {rank_row['W_mean']:.4f} ± {rank_row['W_std']:.4f}")
        print(f"  百分比法: {percent_row['W_mean']:.4f} ± {percent_row['W_std']:.4f}")
        
        print(f"\n纯粉丝偏差 (D): {D_winner} 更优")
        print(f"  排名法: {rank_row['D_mean']:.4f} ± {rank_row['D_std']:.4f}")
        print(f"  百分比法: {percent_row['D_mean']:.4f} ± {percent_row['D_std']:.4f}")
        
        # 综合判断
        winners = [rho_winner, W_winner, D_winner]
        if winners.count("排名法") >= 2:
            print(f"\n[结论] 综合判断：排名法更偏向粉丝投票（{winners.count('排名法')}/3 指标占优）")
        else:
            print(f"\n[结论] 综合判断：百分比法更偏向粉丝投票（{winners.count('百分比法')}/3 指标占优）")
    
    return results_df, summary_df


if __name__ == "__main__":
    # 测试代码
    from data_loader import load_fan_vote_estimates, preprocess_data
    
    df = load_fan_vote_estimates("../fan_vote_estimates.csv")
    df = preprocess_data(df)
    
    results_df, summary_df = compare_all_seasons(df)
    
    # 保存结果
    results_df.to_csv("../输出/表格输出/macro_comparison.csv", index=False, encoding='utf-8-sig')
    summary_df.to_csv("../输出/表格输出/macro_summary.csv", index=False, encoding='utf-8-sig')
    
    print("\n[OK] 结果已保存")
