"""
数据加载与预处理模块
"""
import pandas as pd
import numpy as np
import os


def load_fan_vote_estimates(data_path="fan_vote_estimates.csv"):
    """
    加载粉丝投票估计数据
    
    Parameters:
    -----------
    data_path : str
        数据文件路径
    
    Returns:
    --------
    df : DataFrame
        加载的数据
    """
    print("=" * 60)
    print("加载粉丝投票估计数据...")
    print("=" * 60)
    
    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"[OK] 加载成功，共 {len(df)} 条记录")
    print(f"[OK] 包含 {df['season'].nunique()} 个赛季")
    print(f"[OK] 包含 {df['celebrity_name'].nunique()} 位选手")
    
    return df


def preprocess_data(df):
    """
    预处理数据
    
    Parameters:
    -----------
    df : DataFrame
        原始数据
    
    Returns:
    --------
    df : DataFrame
        预处理后的数据
    """
    print("\n数据预处理...")
    
    # 确保数值列是正确的类型
    numeric_cols = ['judge_rank', 'judge_percent', 'fan_rank', 'fan_votes', 
                    'fan_percent_mean', 'relative_level']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 处理缺失值
    print(f"[OK] 缺失值统计：")
    missing = df[numeric_cols].isnull().sum()
    for col in missing[missing > 0].index:
        print(f"  - {col}: {missing[col]} 个缺失值")
    
    # 对于粉丝百分比为空的记录（使用排名法的赛季），用排名法数据填充
    if df['fan_percent_mean'].isnull().any():
        # 根据粉丝排名估算百分比（简单线性映射）
        mask = df['fan_percent_mean'].isnull() & df['fan_rank'].notnull()
        for idx in df[mask].index:
            season = df.loc[idx, 'season']
            week = df.loc[idx, 'week']
            rank = df.loc[idx, 'fan_rank']
            
            # 获取本周参赛人数
            n_contestants = len(df[(df['season'] == season) & (df['week'] == week)])
            
            # 线性映射：排名1 -> 100/n, 排名n -> 100/n（确保总和为100）
            # 使用更合理的递减模式
            if n_contestants > 1:
                # 排名越小，百分比越高
                df.loc[idx, 'fan_percent_mean'] = 100.0 * (n_contestants - rank + 1) / sum(range(1, n_contestants + 1))
            else:
                df.loc[idx, 'fan_percent_mean'] = 100.0
    
    # 标准化百分比数据到0-100范围
    if 'judge_percent' in df.columns:
        df['judge_percent'] = df['judge_percent'].clip(0, 100)
    if 'fan_percent_mean' in df.columns:
        df['fan_percent_mean'] = df['fan_percent_mean'].clip(0, 100)
    
    print("[OK] 数据预处理完成")
    
    return df


def get_season_data(df, season):
    """
    获取特定赛季的数据
    
    Parameters:
    -----------
    df : DataFrame
        完整数据
    season : int
        赛季编号
    
    Returns:
    --------
    season_df : DataFrame
        该赛季的数据
    """
    return df[df['season'] == season].copy()


def get_contestant_data(df, contestant_name, season):
    """
    获取特定选手在特定赛季的数据
    
    Parameters:
    -----------
    df : DataFrame
        完整数据
    contestant_name : str
        选手姓名
    season : int
        赛季编号
    
    Returns:
    --------
    contestant_df : DataFrame
        该选手的数据
    """
    return df[(df['celebrity_name'] == contestant_name) & 
              (df['season'] == season)].copy()


def create_output_dirs(base_path="输出"):
    """
    创建输出目录结构
    
    Parameters:
    -----------
    base_path : str
        输出根目录
    """
    dirs = [
        os.path.join(base_path, "表格输出"),
        os.path.join(base_path, "可视化"),
        os.path.join(base_path, "分析报告")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"[OK] 输出目录已创建：{base_path}")


if __name__ == "__main__":
    # 测试数据加载
    df = load_fan_vote_estimates("../fan_vote_estimates.csv")
    df = preprocess_data(df)
    print(f"\n数据概览：")
    print(df.head())
    print(f"\n数据统计：")
    print(df.describe())
