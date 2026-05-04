"""
可视化模块
生成所有分析图表
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文显示
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn风格
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# 颜色方案
COLORS = {
    'rank': '#3498db',      # 排名法-蓝色
    'percent': '#e74c3c'    # 百分比法-橙色/红色
}


def plot_fan_bias_comparison(summary_df, output_path="../输出/可视化/fan_bias_comparison.png"):
    """
    绘制粉丝偏向性对比图
    
    Parameters:
    -----------
    summary_df : DataFrame
        跨赛季汇总统计
    output_path : str
        输出路径
    """
    if len(summary_df) == 0:
        print("警告：无数据可绘制")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = [
        ('rho_mean', 'rho_std', 'ρ (Fan Correlation Coefficient)', 'Correlation Coefficient', True),
        ('W_mean', 'W_std', 'W (Fan Contribution Weight)', 'Weight', True),
        ('D_mean', 'D_std', 'D (Pure Fan Bias)', 'Bias', False)
    ]

    for idx, (mean_col, std_col, title, ylabel, higher_is_better) in enumerate(metrics):
        ax = axes[idx]

        # 准备数据
        methods = []
        means = []
        stds = []
        colors = []

        for _, row in summary_df.iterrows():
            method = row['method']
            method_label = 'Rank Method' if method == 'rank' else 'Percentage Method'
            methods.append(method_label)
            means.append(row[mean_col])
            stds.append(row[std_col])
            colors.append(COLORS[method])

        # 绘制柱状图
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, color=colors,
                      edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}\n±{std:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 添加更优指示
        if len(means) == 2:
            if higher_is_better:
                better_idx = 0 if means[0] > means[1] else 1
            else:
                better_idx = 0 if means[0] < means[1] else 1

            # 在更优的柱子上添加星标
            best_bar = bars[better_idx]
            ax.text(best_bar.get_x() + best_bar.get_width()/2.,
                   best_bar.get_height() + stds[better_idx] + 0.05,
                   '★', ha='center', va='bottom', fontsize=20, color='gold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        # 设置y轴范围，留出空间显示标签
        y_max = max([m + s for m, s in zip(means, stds)]) * 1.3
        ax.set_ylim(0, y_max)

    plt.suptitle('Fan Bias Metrics Comparison Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {output_path}")
    plt.close()


def plot_controversial_cases_radar(results_df, output_path="../输出/可视化/controversial_cases_radar.png"):
    """
    绘制争议选手雷达图

    Parameters:
    -----------
    results_df : DataFrame
        争议选手分析结果
    output_path : str
        输出路径
    """
    if len(results_df) == 0:
        print("警告：无数据可绘制")
        return

    # 创建2x2子图（每个选手一个雷达图）
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(results_df.iterrows()):
        if idx >= 4:
            break

        ax = axes[idx]

        # 准备数据（使用标准化的值便于比较）
        # 将排名转换为分数（越小越好 -> 越大越好）
        n = row['n_initial']
        rank_R_score = (n - row['rank_R'] + 1) / n * 100
        rank_P_score = (n - row['rank_P'] + 1) / n * 100
        rank_J_score = (n - row['rank_J'] + 1) / n * 100

        categories = ['Relative Ranking\n(Rank Method)', 'Relative Ranking\n(Percentage Method)',
                     'Fan Rescue Effect\n(Rank Method)', 'Fan Rescue Effect\n(Percentage Method)',
                     'Pure Judge Ranking']

        # 排名法数据
        values_rank = [rank_R_score, 0, row['gamma_R'], 0, rank_J_score]
        # 百分比法数据
        values_percent = [0, rank_P_score, 0, row['gamma_P'], rank_J_score]

        # 闭合雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_rank += values_rank[:1]
        values_percent += values_percent[:1]
        angles += angles[:1]

        # 绘制
        ax.plot(angles, values_rank, 'o-', linewidth=2, label='Rank Method', color=COLORS['rank'])
        ax.fill(angles, values_rank, alpha=0.25, color=COLORS['rank'])

        ax.plot(angles, values_percent, 's-', linewidth=2, label='Percentage Method', color=COLORS['percent'])
        ax.fill(angles, values_percent, alpha=0.25, color=COLORS['percent'])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_title(f"{row['player_name']}\n(Season {row['season']})",
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

    plt.suptitle('Controversial Cases Mechanism Sensitivity Radar Chart', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {output_path}")
    plt.close()


def plot_controversial_cases_scatter(results_df, output_path="../输出/可视化/controversial_cases_scatter.png"):
    """
    绘制争议选手γ对比散点图

    Parameters:
    -----------
    results_df : DataFrame
        争议选手分析结果
    output_path : str
        输出路径
    """
    if len(results_df) == 0:
        print("警告：无数据可绘制")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制对角线（表示γ_R = γ_P）
    max_gamma = max(results_df['gamma_R'].max(), results_df['gamma_P'].max())
    min_gamma = min(results_df['gamma_R'].min(), results_df['gamma_P'].min())
    ax.plot([min_gamma, max_gamma], [min_gamma, max_gamma], 'k--',
            alpha=0.3, linewidth=2, label='Equality Line')

    # 绘制散点
    for _, row in results_df.iterrows():
        ax.scatter(row['gamma_R'], row['gamma_P'], s=300, alpha=0.7,
                  edgecolors='black', linewidth=2)
        ax.annotate(f"{row['player_name']}\n(S{row['season']})",
                   (row['gamma_R'], row['gamma_P']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax.set_xlabel('γ_R (Fan Rescue Effect by Rank Method %)', fontsize=12, fontweight='bold')
    ax.set_ylabel('γ_P (Fan Rescue Effect by Percentage Method %)', fontsize=12, fontweight='bold')
    ax.set_title('Controversial Cases Fan Rescue Effect Comparison\n(γ_R vs γ_P)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # 添加区域标注
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    # 添加象限说明
    ax.text(0.95, 0.05, 'Rank Method More Favorable', transform=ax.transAxes,
           fontsize=11, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor=COLORS['rank'], alpha=0.3))
    ax.text(0.05, 0.95, 'Percentage Method More Favorable', transform=ax.transAxes,
           fontsize=11, ha='left', va='top',
           bbox=dict(boxstyle='round', facecolor=COLORS['percent'], alpha=0.3))

    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {output_path}")
    plt.close()


def plot_rule_impact_heatmap(results_df, output_path="../输出/可视化/rule_impact_heatmap.png"):
    """
    绘制规则影响热力图

    Parameters:
    -----------
    results_df : DataFrame
        规则影响分析结果
    output_path : str
        输出路径
    """
    if len(results_df) == 0:
        print("警告：无数据可绘制")
        return

    # 准备热力图数据
    players = results_df['player_name'].tolist()

    # 选择关键指标
    metrics = [
        'delta_rank_rel_no_rule',
        'delta_rank_rel_with_rule',
        'delta_rank_rel_change',
        'eta_R',
        'eta_P'
    ]

    metric_labels = [
        'ΔRank_rel\n(No Rule)',
        'ΔRank_rel\n(With Rule)',
        'ΔRank_rel\nChange Rate(%)',
        'Elimination Risk η\n(Rank Method%)',
        'Elimination Risk η\n(Percentage Method%)'
    ]

    # 创建数据矩阵
    data = []
    for player in players:
        row_data = []
        player_row = results_df[results_df['player_name'] == player].iloc[0]
        for metric in metrics:
            row_data.append(player_row[metric])
        data.append(row_data)

    data = np.array(data)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 标准化数据用于颜色映射（每列独立标准化）
    data_normalized = data.copy()
    for i in range(data.shape[1]):
        col = data[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 0:
            data_normalized[:, i] = (col - col_min) / (col_max - col_min)

    im = ax.imshow(data_normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # 设置刻度
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(players)))
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
    ax.set_yticklabels([f"{p}\n(S{results_df[results_df['player_name']==p].iloc[0]['season']})"
                        for p in players], fontsize=11, fontweight='bold')

    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # 在每个格子中显示实际数值
    for i in range(len(players)):
        for j in range(len(metrics)):
            value = data[i, j]
            # 根据背景颜色选择文字颜色
            text_color = 'white' if data_normalized[i, j] > 0.5 else 'black'
            text = ax.text(j, i, f'{value:.1f}',
                          ha="center", va="center", color=text_color,
                          fontsize=10, fontweight='bold')

    ax.set_title('Rule Impact Heatmap - Controversial Cases', fontsize=14, fontweight='bold', pad=20)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Value', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {output_path}")
    plt.close()


def plot_season28_comparison(vertical_df, horizontal_df, output_path="../输出/可视化/season28_comparison_bar.png"):
    """
    绘制28季对比柱状图

    Parameters:
    -----------
    vertical_df : DataFrame
        纵向比较结果
    horizontal_df : DataFrame
        横向比较结果
    output_path : str
        输出路径
    """
    if vertical_df is None or horizontal_df is None:
        print("警告：无28季数据可绘制")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 纵向比较
    ax1 = axes[0]
    metrics = vertical_df['metric'].tolist()
    seasons_1_2 = vertical_df['seasons_1_2_avg'].tolist()
    season_28 = vertical_df['season_28'].tolist()

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, seasons_1_2, width, label='Seasons 1-2\n(Rank Method, No Rule)',
                    color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, season_28, width, label='Season 28\n(Rank Method, With Rule)',
                    color=COLORS['rank'], alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title('Vertical Comparison: Rank Method with Rule vs Without Rule', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['ρ', 'W', 'D'], fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # 横向比较
    ax2 = axes[1]
    metrics = horizontal_df['metric'].tolist()
    seasons_3_27 = horizontal_df['seasons_3_27_avg'].tolist()
    season_28 = horizontal_df['season_28'].tolist()

    x = np.arange(len(metrics))

    bars1 = ax2.bar(x - width/2, seasons_3_27, width, label='Seasons 3-27\n(Percentage Method, No Rule)',
                    color=COLORS['percent'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, season_28, width, label='Season 28\n(Rank Method, With Rule)',
                    color=COLORS['rank'], alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.set_title('Horizontal Comparison: Rank Method+Rule vs Percentage Method', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['ρ', 'W', 'D'], fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Season 28 Special Analysis Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("可视化模块测试")
    print("请从main.py运行完整分析流程")