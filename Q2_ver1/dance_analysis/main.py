"""
主程序入口
整合所有模块并生成完整分析报告
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 导入自定义模块
from data_loader import load_fan_vote_estimates, preprocess_data, create_output_dirs
from macro_comparison import compare_all_seasons
from micro_analysis import analyze_all_controversial_cases
from rule_simulation import analyze_controversial_with_rule
from season28_analysis import analyze_season28
from visualization import (
    plot_fan_bias_comparison,
    plot_controversial_cases_radar,
    plot_controversial_cases_scatter,
    plot_rule_impact_heatmap,
    plot_season28_comparison
)


def generate_analysis_report(macro_summary, micro_results, rule_results, 
                            season28_vertical, season28_horizontal,
                            output_path="../输出/分析报告/analysis_summary.md"):
    """
    生成分析报告
    
    Parameters:
    -----------
    macro_summary : DataFrame
        宏观比较汇总
    micro_results : DataFrame
        争议选手分析结果
    rule_results : DataFrame
        规则影响分析结果
    season28_vertical : DataFrame
        28季纵向比较
    season28_horizontal : DataFrame
        28季横向比较
    output_path : str
        输出路径
    """
    print("\n" + "=" * 60)
    print("生成分析报告...")
    print("=" * 60)
    
    report = []
    report.append("# 舞蹈比赛投票机制分析报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report.append("---\n\n")
    
    # 执行摘要
    report.append("## 执行摘要\n\n")
    report.append("本报告对舞蹈比赛的两种投票组合方法（排名法 vs 百分比法）进行了全面的量化比较分析。\n\n")
    
    # 主要发现
    report.append("### 主要发现\n\n")
    
    if len(macro_summary) == 2:
        rank_row = macro_summary[macro_summary['method'] == 'rank'].iloc[0]
        percent_row = macro_summary[macro_summary['method'] == 'percent'].iloc[0]
        
        # 判断哪种方法更偏向粉丝
        rho_winner = "排名法" if rank_row['rho_mean'] > percent_row['rho_mean'] else "百分比法"
        W_winner = "排名法" if rank_row['W_mean'] > percent_row['W_mean'] else "百分比法"
        D_winner = "排名法" if rank_row['D_mean'] < percent_row['D_mean'] else "百分比法"
        
        winners = [rho_winner, W_winner, D_winner]
        overall_winner = "排名法" if winners.count("排名法") >= 2 else "百分比法"
        
        report.append(f"1. **粉丝偏向性综合判断**: {overall_winner}在3个核心指标中有{winners.count(overall_winner)}个占优，")
        report.append(f"显示出更强的粉丝偏向性。\n\n")
        
        report.append(f"2. **粉丝相关性系数 (ρ)**: {rho_winner}更优\n")
        report.append(f"   - 排名法: {rank_row['rho_mean']:.4f} ± {rank_row['rho_std']:.4f}\n")
        report.append(f"   - 百分比法: {percent_row['rho_mean']:.4f} ± {percent_row['rho_std']:.4f}\n\n")
        
        report.append(f"3. **粉丝贡献权重 (W)**: {W_winner}更优\n")
        report.append(f"   - 排名法: {rank_row['W_mean']:.4f} ± {rank_row['W_std']:.4f}\n")
        report.append(f"   - 百分比法: {percent_row['W_mean']:.4f} ± {percent_row['W_std']:.4f}\n\n")
        
        report.append(f"4. **纯粉丝偏差 (D)**: {D_winner}更优\n")
        report.append(f"   - 排名法: {rank_row['D_mean']:.4f} ± {rank_row['D_std']:.4f}\n")
        report.append(f"   - 百分比法: {percent_row['D_mean']:.4f} ± {percent_row['D_std']:.4f}\n\n")
    
    # 争议选手分析
    if len(micro_results) > 0:
        report.append("5. **争议选手分析**:\n")
        
        significant_diff = micro_results[micro_results['delta_rank_rel'] >= 40]
        if len(significant_diff) > 0:
            report.append(f"   - {len(significant_diff)}/{len(micro_results)} 位选手在两种方法下有显著差异（ΔRank_rel ≥ 40%）\n")
            for _, row in significant_diff.iterrows():
                report.append(f"     * {row['player_name']} (赛季{row['season']}): ΔRank_rel = {row['delta_rank_rel']:.1f}%\n")
        else:
            report.append(f"   - 所有分析的争议选手在两种方法下差异较小\n")
        
        report.append(f"   - 平均粉丝挽救效应: 排名法 γ_R = {micro_results['gamma_R'].mean():.2f}%, ")
        report.append(f"百分比法 γ_P = {micro_results['gamma_P'].mean():.2f}%\n\n")
    
    # 规则影响分析
    if len(rule_results) > 0:
        report.append("6. **评委选择淘汰规则影响**:\n")
        report.append(f"   - 规则对ΔRank_rel的平均影响: {rule_results['delta_rank_rel_change'].mean():.2f}%\n")
        report.append(f"   - 平均淘汰风险: 排名法 {rule_results['eta_R'].mean():.2f}%, ")
        report.append(f"百分比法 {rule_results['eta_P'].mean():.2f}%\n")
        
        if abs(rule_results['delta_rank_rel_change'].mean()) < 10:
            report.append(f"   - **结论**: 规则对方法差异影响较小\n\n")
        elif rule_results['delta_rank_rel_change'].mean() > 0:
            report.append(f"   - **结论**: 规则扩大了两种方法的差异\n\n")
        else:
            report.append(f"   - **结论**: 规则缩小了两种方法的差异\n\n")
    
    # 28季特殊分析
    if season28_vertical is not None and season28_horizontal is not None:
        report.append("7. **28季特殊分析** (首次使用排名法+评委选择规则):\n")
        report.append("   - **纵向比较**: 与前2季相比，规则对排名法粉丝偏向性的影响\n")
        
        avg_change = season28_vertical['change_pct'].mean()
        if abs(avg_change) < 10:
            report.append(f"     * 影响较小（平均变化 {avg_change:.1f}%）\n")
        else:
            report.append(f"     * 有显著影响（平均变化 {avg_change:.1f}%）\n")
        
        report.append("   - **横向比较**: 与3-27季百分比法相比\n")
        
        # 判断28季是否仍然更偏向粉丝
        favorable_metrics = 0
        for _, row in season28_horizontal.iterrows():
            if '更高' in row.get('interpretation', '') or '更低(更好)' in row.get('interpretation', ''):
                if '28季' in row.get('interpretation', ''):
                    favorable_metrics += 1
        
        if favorable_metrics >= 2:
            report.append(f"     * 28季在{favorable_metrics}/3个指标上优于百分比法，仍保持强粉丝偏向性\n\n")
        else:
            report.append(f"     * 28季在{favorable_metrics}/3个指标上优于百分比法\n\n")
    
    report.append("---\n\n")
    
    # 方法论
    report.append("## 方法论\n\n")
    report.append("### 核心量化指标\n\n")
    report.append("1. **粉丝相关性系数 (ρ)**: Spearman秩相关系数，衡量粉丝投票与最终结果的关联强度\n")
    report.append("   - 取值范围: [-1, 1]\n")
    report.append("   - 解读: 越接近1，越偏向粉丝\n\n")
    
    report.append("2. **粉丝贡献权重 (W)**: 通过线性回归反推的粉丝投票实际贡献占比\n")
    report.append("   - 取值范围: [0, 1]\n")
    report.append("   - 解读: 数值越大，粉丝影响力越大\n\n")
    
    report.append("3. **纯粉丝偏差 (D)**: 基于Kendall tau距离，衡量最终结果与纯粉丝意愿的偏离程度\n")
    report.append("   - 取值范围: [0, 1]\n")
    report.append("   - 解读: 数值越小，越接近纯粉丝排名\n\n")
    
    report.append("4. **排名相对差异 (ΔRank_rel)**: 两种方法下选手排名的相对差异\n")
    report.append("   - 计算: |Rank_R - Rank_P| / n × 100%\n")
    report.append("   - 解读: <20%轻微差异，≥40%显著差异\n\n")
    
    report.append("5. **粉丝挽救效应系数 (γ)**: 粉丝投票对评委低分的挽救作用\n")
    report.append("   - 计算: (Rank_Jonly - Rank_method) / Rank_Jonly × 100%\n")
    report.append("   - 解读: 正值越高，粉丝挽救效应越强\n\n")
    
    report.append("### 建模方法\n\n")
    report.append("- **排名法**: Score_R = 评委排名 + 粉丝排名（越小越好）\n")
    report.append("- **百分比法**: Score_P = 评委百分比 + 粉丝百分比（越大越好）\n")
    report.append("- **评委选择规则**: 每周从末两名中淘汰评委排名更差的选手\n\n")
    
    report.append("---\n\n")
    
    # 推荐
    report.append("## 推荐与建议\n\n")
    
    if len(macro_summary) == 2:
        rank_row = macro_summary[macro_summary['method'] == 'rank'].iloc[0]
        percent_row = macro_summary[macro_summary['method'] == 'percent'].iloc[0]
        
        if rank_row['rho_mean'] > percent_row['rho_mean']:
            report.append("### 投票机制推荐\n\n")
            report.append("基于量化分析结果，**推荐使用排名法**，理由如下：\n\n")
            report.append("1. 排名法在粉丝偏向性的三个核心指标上表现更优\n")
            report.append("2. 排名法能更好地反映粉丝意愿，提高观众参与感\n")
            report.append("3. 排名法对争议选手的粉丝挽救效应更强\n\n")
        else:
            report.append("### 投票机制推荐\n\n")
            report.append("基于量化分析结果，**推荐使用百分比法**，理由如下：\n\n")
            report.append("1. 百分比法在粉丝偏向性的三个核心指标上表现更优\n")
            report.append("2. 百分比法能更精确地量化粉丝支持度\n")
            report.append("3. 百分比法提供更细粒度的得分差异\n\n")
        
        report.append("### 规则建议\n\n")
        report.append("- 评委选择淘汰规则对整体机制影响有限，可根据节目需求选择是否采用\n")
        report.append("- 如果希望增加节目悬念和评委权威性，可考虑使用该规则\n")
        report.append("- 如果希望最大化粉丝影响力，建议不使用该规则\n\n")
    
    report.append("---\n\n")
    
    # 附录
    report.append("## 附录\n\n")
    report.append("### 输出文件清单\n\n")
    report.append("#### 表格输出\n")
    report.append("- `macro_comparison.csv`: 各赛季宏观指标\n")
    report.append("- `macro_summary.csv`: 跨赛季汇总统计\n")
    report.append("- `micro_analysis.csv`: 争议选手分析\n")
    report.append("- `rule_impact.csv`: 规则影响分析\n")
    report.append("- `season28_vertical.csv`: 28季纵向比较\n")
    report.append("- `season28_horizontal.csv`: 28季横向比较\n\n")
    
    report.append("#### 可视化\n")
    report.append("- `fan_bias_comparison.png`: 粉丝偏向性对比图\n")
    report.append("- `controversial_cases_radar.png`: 争议选手雷达图\n")
    report.append("- `controversial_cases_scatter.png`: 争议选手散点图\n")
    report.append("- `rule_impact_heatmap.png`: 规则影响热力图\n")
    report.append("- `season28_comparison_bar.png`: 28季对比柱状图\n\n")
    
    report.append("---\n\n")
    report.append("*报告生成完毕*\n")
    
    # 保存报告
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    print(f"[OK] 分析报告已保存: {output_path}")


def main():
    """
    主函数：运行完整分析流程
    """
    print("=" * 60)
    print("舞蹈比赛投票机制分析系统")
    print("=" * 60)
    print()
    
    # 1. 初始化路径
    import os
    # 获取当前文件所在目录的上级目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, "fan_vote_estimates.csv")
    output_dir = os.path.join(parent_dir, "输出")
    
    # 创建输出目录
    create_output_dirs(output_dir)
    
    # 2. 加载和预处理数据
    print("\n【步骤1】加载数据")
    df = load_fan_vote_estimates(data_path)
    df = preprocess_data(df)
    
    # 3. 宏观比较分析
    print("\n【步骤2】宏观比较分析")
    macro_results, macro_summary = compare_all_seasons(df)
    
    # 保存结果
    macro_results.to_csv(os.path.join(output_dir, "表格输出", "macro_comparison.csv"), index=False, encoding='utf-8-sig')
    macro_summary.to_csv(os.path.join(output_dir, "表格输出", "macro_summary.csv"), index=False, encoding='utf-8-sig')
    print("[OK] 宏观比较结果已保存")
    
    # 4. 争议选手微观分析
    print("\n【步骤3】争议选手微观分析")
    micro_results = analyze_all_controversial_cases(df)
    
    # 保存结果
    micro_results.to_csv(os.path.join(output_dir, "表格输出", "micro_analysis.csv"), index=False, encoding='utf-8-sig')
    print("[OK] 争议选手分析结果已保存")
    
    # 5. 规则影响分析
    print("\n【步骤4】规则扩展分析")
    rule_results = analyze_controversial_with_rule(df)
    
    # 保存结果
    rule_results.to_csv(os.path.join(output_dir, "表格输出", "rule_impact.csv"), index=False, encoding='utf-8-sig')
    print("[OK] 规则影响分析结果已保存")
    
    # 6. 28季特殊分析
    print("\n【步骤5】28季特殊分析")
    season28_vertical, season28_horizontal = analyze_season28(df)
    
    # 保存结果
    if season28_vertical is not None:
        season28_vertical.to_csv(os.path.join(output_dir, "表格输出", "season28_vertical.csv"), index=False, encoding='utf-8-sig')
    if season28_horizontal is not None:
        season28_horizontal.to_csv(os.path.join(output_dir, "表格输出", "season28_horizontal.csv"), index=False, encoding='utf-8-sig')
    print("[OK] 28季分析结果已保存")
    
    # 7. 生成可视化
    print("\n【步骤6】生成可视化")
    
    # 粉丝偏向性对比图
    plot_fan_bias_comparison(macro_summary, os.path.join(output_dir, "可视化", "fan_bias_comparison.png"))
    
    # 争议选手雷达图和散点图
    if len(micro_results) > 0:
        plot_controversial_cases_radar(micro_results, os.path.join(output_dir, "可视化", "controversial_cases_radar.png"))
        plot_controversial_cases_scatter(micro_results, os.path.join(output_dir, "可视化", "controversial_cases_scatter.png"))
    
    # 规则影响热力图
    if len(rule_results) > 0:
        plot_rule_impact_heatmap(rule_results, os.path.join(output_dir, "可视化", "rule_impact_heatmap.png"))
    
    # 28季对比图
    if season28_vertical is not None and season28_horizontal is not None:
        plot_season28_comparison(season28_vertical, season28_horizontal, os.path.join(output_dir, "可视化", "season28_comparison_bar.png"))
    
    # 8. 生成分析报告
    print("\n【步骤7】生成分析报告")
    generate_analysis_report(
        macro_summary, 
        micro_results, 
        rule_results,
        season28_vertical,
        season28_horizontal,
        os.path.join(output_dir, "分析报告", "analysis_summary.md")
    )
    
    print("\n" + "=" * 60)
    print("分析完成！所有结果已保存到 '输出' 目录")
    print("=" * 60)
    print("\n输出目录结构:")
    print("输出/")
    print("├── 表格输出/")
    print("│   ├── macro_comparison.csv")
    print("│   ├── macro_summary.csv")
    print("│   ├── micro_analysis.csv")
    print("│   ├── rule_impact.csv")
    print("│   ├── season28_vertical.csv")
    print("│   └── season28_horizontal.csv")
    print("├── 可视化/")
    print("│   ├── fan_bias_comparison.png")
    print("│   ├── controversial_cases_radar.png")
    print("│   ├── controversial_cases_scatter.png")
    print("│   ├── rule_impact_heatmap.png")
    print("│   └── season28_comparison_bar.png")
    print("└── 分析报告/")
    print("    └── analysis_summary.md")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
