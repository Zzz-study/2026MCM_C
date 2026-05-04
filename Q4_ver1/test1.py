"""
舞蹈进化生态系统模型 - 完整实现
Dance
Evolution
Ecosystem
Model - Complete
Implementation
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class Player:
    """
选手类 - 表示一个舞蹈竞赛选手
"""

    def __init__(self, player_id: str, name: str, season: int):
        self.id = player_id
        self.name = name
        self.season = season

        # 核心属性
        self.tech_score = 0.5  # 技术基因评分 [0,1]
        self.adaptation = 0.5   # 环境适应性 [0,1]
        self.novelty = 0.0      # 创新性 [0,1]
        self.experience = 0     # 经验 (0或1)
        self.trend = 0.0        # 进化趋势

        # 社交网络
        self.neighbors = []
        self.industry = ""
        self.partner = ""

        # 历史数据
        self.judge_scores_history = []
        self.survival_index_history = []
        self.week_data = {}

    def calculate_survival_index(self, weights: Dict[str, float]) -> float:
        """
计算生存指数
"""
        survival_index = (
            weights['tech'] * self.tech_score +
            weights['fan'] * self.adaptation +
            weights['trend'] * self.trend
        )
        return survival_index

    def update_tech_score(self, judge_scores: List[float]):
        """
更新技术基因评分
"""
        if not judge_scores:
            return

        # 标准化到[0,1]
        normalized_scores = [s / 40.0 for s in judge_scores]  # 假设满分40

        # 计算多维度基因
        rhythm_gene = np.mean([normalized_scores[0], normalized_scores[2]]) if len(normalized_scores) >= 3 else np.mean(normalized_scores)
        elegance_gene = np.mean([normalized_scores[1], normalized_scores[3]]) if len(normalized_scores) >= 4 else np.mean(normalized_scores)
        creativity_gene = 1 - (np.std(normalized_scores) / 0.25) if len(normalized_scores) > 1 else 0.5
        sync_gene = 1 - (np.var(normalized_scores) / 0.25) if len(normalized_scores) > 1 else 0.5

        # 加权平均
        self.tech_score = np.clip(
            0.3 * rhythm_gene + 0.3 * elegance_gene + 0.2 * creativity_gene + 0.2 * sync_gene,
            0, 1
        )

    def update_novelty(self, judge_scores: List[float]):
        """
更新创新性
"""
        if len(judge_scores) > 1:
            self.novelty = np.clip(np.std(judge_scores) / 5.0, 0, 1)
        else:
            self.novelty = 0

    def update_trend(self):
        """
更新进化趋势
"""
        if len(self.judge_scores_history) >= 2:
            recent_scores = self.judge_scores_history[-2:]
            if len(recent_scores[0]) > 0 and len(recent_scores[1]) > 0:
                progress_speed = (np.mean(recent_scores[-1]) - np.mean(recent_scores[-2])) / 5.0
                innovation = self.novelty
                self.trend = np.clip(progress_speed + 0.3 * innovation, -1, 1)
        else:
            self.trend = 0


class FairnessMetrics:
    """
公平性指标
"""

    @staticmethod
    def controversy_index(fan_ranks: np.ndarray, judge_ranks: np.ndarray) -> float:
        """
争议指数 = 粉丝排名与评委排名差异的平均值
"""
        return np.mean(np.abs(fan_ranks - judge_ranks))

    @staticmethod
    def balance_index(top_tech_survival_rate: float, top_fan_survival_rate: float) -> float:
        """
平衡指数 = 1 - | 技术前20 % 晋级率 - 粉丝前20 % 晋级率 | """
        return 1 - abs(top_tech_survival_rate - top_fan_survival_rate)

    @staticmethod
    def meritocracy_score(eliminated_tech_ranks: List[float], total_players: int) -> float:
        """
精英保护分数 = 1 - (被淘汰选手中技术排名前30 % 的比例)
"""
        if len(eliminated_tech_ranks) == 0:
            return 1.0
        tech_top_30 = [r for r in eliminated_tech_ranks if r <= 0.3]
        return 1 - (len(tech_top_30) / len(eliminated_tech_ranks))


class EntertainmentMetrics:
    """
娱乐性指标
"""

    @staticmethod
    def suspense_index(survival_scores: List[float]) -> float:
        """
悬念指数 = 淘汰候选池选手生存指数的变异系数
"""
        if len(survival_scores) < 2:
            return 0
        mean_score = np.mean(survival_scores)
        if mean_score == 0:
            return 0
        return np.std(survival_scores) / mean_score

    @staticmethod
    def surprise_index(eliminated_rank: int, candidate_ranks: List[int]) -> float:
        """
意外指数 = 淘汰选手在候选池中的排名
"""
        if eliminated_rank not in candidate_ranks or len(candidate_ranks) == 0:
            return 0
        rank_position = candidate_ranks.index(eliminated_rank)
        return 1 - (rank_position / len(candidate_ranks))

    @staticmethod
    def engagement_index(fan_vote_volatility: List[float]) -> float:
        """
参与度指数 = 粉丝投票波动率
"""
        if len(fan_vote_volatility) < 2:
            return 0
        mean_vol = np.mean(fan_vote_volatility)
        if mean_vol == 0:
            return 0
        return np.std(fan_vote_volatility) / mean_vol


class ControlledRandomEvents:
    """
可控随机事件系统
"""

    def __init__(self, enabled: bool = True, max_impact: float = 0.1):
        self.enabled = enabled
        self.max_impact = max_impact
        self.events_log = []

    def trigger_event(self, week: int, players: List[Player],
                     survival_indices: Dict[str, float]) -> Tuple[List[Player], Dict]:
        """
触发随机事件
"""
        if not self.enabled or random.random() > 0.1:
            return players, {"type": "无事件", "week": week}

        event_type = random.choice([
            'genetic_drift',
            'adaptive_radiation',
            'survival_crisis',
            'symbiosis'
        ])

        event_result = {
            'type': event_type,
            'week': week,
            'affected_players': [],
            'impact': 0,
            'description': ''
        }

        if event_type == 'genetic_drift':
            # 基因漂变
            player = random.choice(players)
            drift = random.uniform(-self.max_impact, self.max_impact)
            player.tech_score = np.clip(player.tech_score * (1 + drift), 0, 1)
            event_result['impact'] = drift
            event_result['affected_players'] = [player.id]
            event_result['description'] = f'基因漂变: {player.name}技术分变化{drift*100:.1f}%'

        elif event_type == 'adaptive_radiation':
            # 适应性辐射
            player = random.choice(players)
            radiation = random.uniform(0, self.max_impact)
            player.adaptation = np.clip(player.adaptation * (1 + radiation), 0, 1)
            event_result['impact'] = radiation
            event_result['affected_players'] = [player.id]
            event_result['description'] = f'适应性辐射: {player.name}适应度提升{radiation*100:.1f}%'

        elif event_type == 'survival_crisis':
            # 生存危机（扩大淘汰池）
            event_result['impact'] = 1
            event_result['description'] = '生存危机: 本周淘汰池扩大至3人'

        elif event_type == 'symbiosis':
            # 共生关系（同舞伴加成）
            if len(players) >= 2:
                player1, player2 = random.sample(players, 2)
                bonus = random.uniform(0, self.max_impact)
                player1.tech_score = np.clip(player1.tech_score * (1 + bonus), 0, 1)
                player2.tech_score = np.clip(player2.tech_score * (1 + bonus), 0, 1)
                event_result['impact'] = bonus
                event_result['affected_players'] = [player1.id, player2.id]
                event_result['description'] = f'共生关系: {player1.name}和{player2.name}互相加成{bonus*100:.1f}%'

        # 安全检查
        self._safety_check(players, event_result)

        self.events_log.append(event_result)
        return players, event_result

    def _safety_check(self, players: List[Player], event_result: Dict):
        """
安全机制：保护优秀选手
"""
        tech_scores = [p.tech_score for p in players]
        top_10_threshold = np.percentile(tech_scores, 90)

        for player_id in event_result['affected_players']:
            player = next((p for p in players if p.id == player_id), None)
            if player and player.tech_score >= top_10_threshold and event_result['impact'] < 0:
                # 负面影响减半
                event_result['impact'] /= 2
                event_result['safety_override'] = True


class SimplifiedDanceEcosystem:
    """
简化版舞蹈生态系统模型 - 只有2个核心调节参数
"""

    def __init__(self):
        # 固定参数
        self.fixed_params = {
            'D': 0.5,           # 扩散系数
            'λ': 0.3,           # 创新奖励
            'θ_novelty': 0.7,   # 创新阈值
            'mutation_rate': 0.1  # 随机事件概率
        }

        # 核心调节滑块
        self.sliders = {
            'professional_entertainment_balance': 0.5,  # 0=纯专业, 1=纯娱乐
            'suspense_intensity': 0.5,  # 0=低悬念, 1=高悬念
        }

        # 随机事件系统
        self.random_events = ControlledRandomEvents(enabled=True, max_impact=0.1)

        # 结果存储
        self.players: List[Player] = []
        self.weekly_results = []
        self.fairness_history = []
        self.entertainment_history = []

    def calculate_weights(self, week: int, total_weeks: int) -> Dict[str, float]:
        """
根据平衡滑块计算动态权重
"""
        balance = self.sliders['professional_entertainment_balance']

        # 基础权重（阶段性变化）
        progress = week / total_weeks
        if progress <= 1/3:
            base_tech, base_fan = 0.6, 0.3  # 早期重专业
        elif progress <= 2/3:
            base_tech, base_fan = 0.45, 0.45  # 中期平衡
        else:
            base_tech, base_fan = 0.3, 0.6  # 后期重娱乐

        # 根据滑块调整
        tech_weight = base_tech + (1 - balance) * 0.2
        fan_weight = base_fan + balance * 0.2

        # 归一化
        total = tech_weight + fan_weight + 0.1
        return {
            'tech': tech_weight / total,
            'fan': fan_weight / total,
            'trend': 0.1 / total
        }

    def calculate_selection_pressure(self) -> float:
        """
根据悬念强度滑块计算选择压力
"""
        suspense = self.sliders['suspense_intensity']
        # 悬念强度高 → 选择压力小 → 淘汰更随机
        return 1.2 - suspense * 0.7  # 范围[0.5, 1.2]

    def load_data(self, features_path: str, processed_path: str, fan_votes_path: str):
        """
加载数据并构建选手对象
"""
        print("正在加载数据...")

        # 读取数据
        df_features = pd.read_excel(features_path)
        df_processed = pd.read_excel(processed_path)

        try:
            df_fan_votes = pd.read_excel(fan_votes_path)
        except:
            # 如果Excel读取失败，创建模拟数据
            print("警告: 无法读取fan_vote_estimates.xlsx，使用模拟数据")
            df_fan_votes = pd.DataFrame({
                'season': df_processed['season'],
                'week': df_processed['week'],
                'celebrity_name': df_processed['celebrity_name'],
                'estimated_fan_votes': np.random.uniform(1000, 10000, len(df_processed))
            })

        # 合并数据
        df_merged = pd.merge(
            df_processed,
            df_fan_votes,
            on=['season', 'week', 'celebrity_name'],
            how='left'
        )

        # 填充缺失值
        if 'estimated_fan_votes' in df_merged.columns:
            df_merged['estimated_fan_votes'].fillna(
                df_merged.groupby('season')['estimated_fan_votes'].transform('mean'),
                inplace=True
            )
        else:
            df_merged['estimated_fan_votes'] = 5000  # 默认值

        # 提取选手特征
        self._extract_features(df_merged)

        print(f"成功加载 {len(self.players)} 名选手数据")

        return df_merged

    def _extract_features(self, df: pd.DataFrame):
        """
从数据中提取选手特征
"""
        player_dict = {}

        for _, row in df.iterrows():
            player_id = f"{row['celebrity_name']}_{row['season']}"

            if player_id not in player_dict:
                player = Player(
                    player_id=player_id,
                    name=row['celebrity_name'],
                    season=row['season']
                )
                player.industry = row.get('celebrity_industry', 'Unknown')
                player.partner = row.get('ballroom_partner', 'Unknown')
                player.experience = 1 if row.get('all_star_season', False) else 0

                player_dict[player_id] = player

            player = player_dict[player_id]
            week = row['week']

            # 存储周数据
            if row['in_competition']:
                judge_score = row.get('total_judge_score', 0)
                fan_votes = row.get('estimated_fan_votes', 5000)

                player.week_data[week] = {
                    'judge_score': judge_score,
                    'fan_votes': fan_votes,
                    'in_competition': True
                }

        # 处理每个选手的历史数据
        for player in player_dict.values():
            weeks = sorted(player.week_data.keys())

            for week in weeks:
                if player.week_data[week]['in_competition']:
                    judge_score = player.week_data[week]['judge_score']
                    fan_votes = player.week_data[week]['fan_votes']

                    # 更新技术评分
                    player.update_tech_score([judge_score])
                    player.judge_scores_history.append([judge_score])

                    # 更新适应性（标准化粉丝投票）
                    player.adaptation = np.clip(fan_votes / 10000.0, 0, 1)

                    # 更新创新性
                    player.update_novelty([judge_score])

                    # 更新趋势
                    player.update_trend()

        self.players = list(player_dict.values())

    def simulate_season(self, season_id: int, total_weeks: int = 10) -> Dict:
        """
模拟一个完整赛季
"""
        print(f"\n=== 模拟第 {season_id} 季 ===")

        # 筛选该季选手
        season_players = [p for p in self.players if p.season == season_id]

        if len(season_players) == 0:
            print(f"警告: 第{season_id}季没有选手数据")
            return {}

        results = {
            'season': season_id,
            'weekly_eliminations': [],
            'final_ranking': [],
            'fairness_metrics': {},
            'entertainment_metrics': {},
            'events_log': []
        }

        active_players = season_players.copy()

        for week in range(1, total_weeks + 1):
            if len(active_players) <= 2:
                break

            print(f"\n第 {week} 周:")

            # 计算权重
            weights = self.calculate_weights(week, total_weeks)
            print(f"权重: 技术={weights['tech']:.2f}, 粉丝={weights['fan']:.2f}, 趋势={weights['trend']:.2f}")

            # 计算生存指数
            survival_indices = {}
            for player in active_players:
                si = player.calculate_survival_index(weights)
                player.survival_index_history.append(si)
                survival_indices[player.id] = si

            # 触发随机事件
            active_players, event = self.random_events.trigger_event(
                week, active_players, survival_indices
            )
            results['events_log'].append(event)

            if event['type'] != '无事件':
                print(f"随机事件: {event.get('description', event['type'])}")

            # 确定淘汰候选池
            sorted_players = sorted(
                active_players,
                key=lambda p: survival_indices[p.id]
            )

            # 根据悬念强度确定候选池大小
            pool_size = 2 if event.get('type') != 'survival_crisis' else 3
            candidate_pool = sorted_players[:min(pool_size, len(sorted_players))]

            # 选择压力影响淘汰概率
            selection_pressure = self.calculate_selection_pressure()

            # 计算淘汰概率（生存指数越低，淘汰概率越高）
            pool_survival_scores = [survival_indices[p.id] for p in candidate_pool]

            if selection_pressure > 1.0:
                # 高选择压力 - 确定性淘汰
                eliminated = candidate_pool[0]
            else:
                # 低选择压力 - 概率性淘汰
                min_score = min(pool_survival_scores)
                max_score = max(pool_survival_scores) if len(pool_survival_scores) > 1 else min_score + 0.1

                # 归一化并反转（分数低的淘汰概率高）
                if max_score > min_score:
                    probs = [(max_score - s) / (max_score - min_score) for s in pool_survival_scores]
                else:
                    probs = [1.0 / len(candidate_pool)] * len(candidate_pool)

                # 归一化概率
                total_prob = sum(probs)
                probs = [p / total_prob for p in probs]

                # 随机选择
                eliminated = np.random.choice(candidate_pool, p=probs)

            print(f"淘汰选手: {eliminated.name} (生存指数: {survival_indices[eliminated.id]:.3f})")
            print(f"候选池: {[p.name for p in candidate_pool]}")

            # 记录淘汰结果
            results['weekly_eliminations'].append({
                'week': week,
                'eliminated': eliminated.name,
                'survival_index': survival_indices[eliminated.id],
                'candidate_pool': [p.name for p in candidate_pool],
                'pool_survival_indices': [survival_indices[p.id] for p in candidate_pool]
            })

            # 移除淘汰选手
            active_players.remove(eliminated)

            # 计算本周娱乐性指标
            entertainment = {
                'suspense': EntertainmentMetrics.suspense_index(pool_survival_scores),
                'week': week
            }
            results['entertainment_metrics'][week] = entertainment

        # 最终排名
        if active_players:
            final_sorted = sorted(
                active_players,
                key=lambda p: p.survival_index_history[-1] if p.survival_index_history else 0,
                reverse=True
            )

            for rank, player in enumerate(final_sorted, 1):
                results['final_ranking'].append({
                    'rank': rank,
                    'name': player.name,
                    'final_survival_index': player.survival_index_history[-1] if player.survival_index_history else 0
                })

        print(f"\n最终排名:")
        for item in results['final_ranking']:
            print(f"{item['rank']}. {item['name']} (生存指数: {item['final_survival_index']:.3f})")

        return results

    def set_slider(self, balance: float = None, suspense: float = None):
        """
设置滑块参数
"""
        if balance is not None:
            self.sliders['professional_entertainment_balance'] = np.clip(balance, 0, 1)
            print(f"专业-娱乐平衡滑块设置为: {balance:.2f}")

        if suspense is not None:
            self.sliders['suspense_intensity'] = np.clip(suspense, 0, 1)
            print(f"悬念强度滑块设置为: {suspense:.2f}")


class ProducerControlPanel:
    """
制片人控制面板
"""

    def __init__(self, model: SimplifiedDanceEcosystem):
        self.model = model

        self.presets = {
            '专业优先模式': {'balance': 0.2, 'suspense': 0.3},
            '娱乐优先模式': {'balance': 0.8, 'suspense': 0.7},
            '平衡模式': {'balance': 0.5, 'suspense': 0.5},
            '高悬念模式': {'balance': 0.5, 'suspense': 0.9}
        }

    def apply_preset(self, preset_name: str):
        """
应用预设模式
"""
        if preset_name not in self.presets:
            print(f"错误: 预设'{preset_name}'不存在")
            print(f"可用预设: {list(self.presets.keys())}")
            return

        preset = self.presets[preset_name]
        self.model.set_slider(
            balance=preset['balance'],
            suspense=preset['suspense']
        )
        print(f"✓ 已应用预设: {preset_name}")

    def show_presets(self):
        """
显示所有预设
"""
        print("\n可用预设模式:")
        print("-" * 50)
        for name, params in self.presets.items():
            print(f"{name}:")
            print(f"  - 专业-娱乐平衡: {params['balance']:.2f}")
            print(f"  - 悬念强度: {params['suspense']:.2f}")
        print("-" * 50)


def visualize_results(results: Dict, save_path: str = None):
    """
    可视化结果，将两张图片分别绘制在不同画布上
    """
    if not results:
        print("没有结果可以可视化")
        return

    # 第一张图：淘汰顺序
    plt.figure(figsize=(10, 8))

    eliminations = results['weekly_eliminations']
    weeks = [e['week'] for e in eliminations]
    survival_indices = [e['survival_index'] for e in eliminations]
    names = [e['eliminated'] for e in eliminations]

    colors = plt.cm.RdYlGn([s for s in survival_indices])
    bars = plt.barh(range(len(names)), survival_indices, color=colors)
    plt.yticks(range(len(names)), [f"Week{w}:{n}" for w, n in zip(weeks, names)], fontsize=8)
    plt.xlabel('Survival Index', fontsize=10)
    plt.title('Season 1 Simulation Results: Eliminated Contestants and Their Survival Index',
              fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()

    if save_path:
        # 保存第一张图
        plot1_path = save_path.replace('.png', '_plot1.png') if save_path.endswith('.png') else save_path + '_plot1.png'
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表1已保存至: {plot1_path}")

    plt.show()

    # 第二张图：候选池悬念度
    plt.figure(figsize=(10, 6))

    entertainment_weeks = []
    suspense_scores = []

    for week, metrics in results['entertainment_metrics'].items():
        entertainment_weeks.append(week)
        suspense_scores.append(metrics['suspense'])

    if entertainment_weeks:
        plt.plot(entertainment_weeks, suspense_scores, marker='o', linewidth=2, markersize=6)
        plt.fill_between(entertainment_weeks, suspense_scores, alpha=0.3)
        plt.xlabel('Week', fontsize=10)
        plt.ylabel('Suspense Index', fontsize=10)
        plt.title('Season 1 Simulation Results: Weekly Suspense Index Changes',
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if save_path:
            # 保存第二张图
            plot2_path = save_path.replace('.png', '_plot2.png') if save_path.endswith(
                '.png') else save_path + '_plot2.png'
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表2已保存至: {plot2_path}")

        plt.show()

    # 3. 最终排名
    # ax3 = axes[1, 0]
    # if results['final_ranking']:
    #     final_names = [f"{r['rank']}. {r['name']}" for r in results['final_ranking']]
    #     final_scores = [r['final_survival_index'] for r in results['final_ranking']]
    #
    #     colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(final_names)))
    #     ax3.barh(range(len(final_names)), final_scores, color=colors)
    #     ax3.set_yticks(range(len(final_names)))
    #     ax3.set_yticklabels(final_names, fontsize=9)
    #     ax3.set_xlabel('最终生存指数', fontsize=10)
    #     ax3.set_title('最终排名', fontsize=12, fontweight='bold')
    #     ax3.invert_yaxis()

    # 4. 随机事件统计
    # ax4 = axes[1, 1]
    # events = results['events_log']
    # event_types = {}
    #
    # for event in events:
    #     event_type = event['type']
    #     if event_type != '无事件':
    #         event_types[event_type] = event_types.get(event_type, 0) + 1
    #
    # if event_types:
    #     event_names = list(event_types.keys())
    #     event_counts = list(event_types.values())
    #
    #     # 中文翻译
    #     event_names_cn = {
    #         'genetic_drift': '基因漂变',
    #         'adaptive_radiation': '适应性辐射',
    #         'survival_crisis': '生存危机',
    #         'symbiosis': '共生关系'
    #     }
    #
    #     display_names = [event_names_cn.get(name, name) for name in event_names]
    #
    #     colors = plt.cm.Set3(np.linspace(0, 1, len(event_names)))
    #     ax4.pie(event_counts, labels=display_names, autopct='%1.1f%%',
    #            colors=colors, startangle=90)
    #     ax4.set_title('随机事件分布', fontsize=12, fontweight='bold')
    # else:
    #     ax4.text(0.5, 0.5, '本季无随机事件',
    #             ha='center', va='center', fontsize=12)
    #     ax4.set_xlim(0, 1)
    #     ax4.set_ylim(0, 1)
    #     ax4.axis('off')
    #
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     print(f"✓ 图表已保存至: {save_path}")
    #
    # plt.show()


def compare_methods(model: SimplifiedDanceEcosystem, season_id: int):
    """
对比三种方法：排名法、百分比法、新模型
"""
    print(f"\n{'='*60}")
    print(f"对比分析: 第 {season_id} 季")
    print(f"{'='*60}\n")

    # 新模型（平衡模式）
    print("1. 新模型（平衡模式）")
    model.set_slider(balance=0.5, suspense=0.5)
    results_balanced = model.simulate_season(season_id)

    # 新模型（专业优先）
    print("\n2. 新模型（专业优先模式）")
    model.set_slider(balance=0.2, suspense=0.3)
    results_professional = model.simulate_season(season_id)

    # 新模型（娱乐优先）
    print("\n3. 新模型（娱乐优先模式）")
    model.set_slider(balance=0.8, suspense=0.7)
    results_entertainment = model.simulate_season(season_id)

    # 对比表格
    print(f"\n{'='*60}")
    print("对比总结")
    print(f"{'='*60}")

    print(f"\n{'模式':<15} {'冠军':<20} {'亚军':<20}")
    print("-" * 60)

    if results_balanced.get('final_ranking'):
        print(f"{'平衡模式':<15} {results_balanced['final_ranking'][0]['name']:<20} {results_balanced['final_ranking'][1]['name'] if len(results_balanced['final_ranking']) > 1 else 'N/A':<20}")

    if results_professional.get('final_ranking'):
        print(f"{'专业优先':<15} {results_professional['final_ranking'][0]['name']:<20} {results_professional['final_ranking'][1]['name'] if len(results_professional['final_ranking']) > 1 else 'N/A':<20}")

    if results_entertainment.get('final_ranking'):
        print(f"{'娱乐优先':<15} {results_entertainment['final_ranking'][0]['name']:<20} {results_entertainment['final_ranking'][1]['name'] if len(results_entertainment['final_ranking']) > 1 else 'N/A':<20}")

    print(f"\n{'='*60}\n")

    return {
        'balanced': results_balanced,
        'professional': results_professional,
        'entertainment': results_entertainment
    }


def main():
    """
主函数
"""
    print("="*70)
    print(" " * 15 + "舞蹈进化生态系统模型")
    print(" " * 10 + "Dance Evolution Ecosystem Model")
    print("="*70 + "\n")

    # 创建模型
    model = SimplifiedDanceEcosystem()

    # 加载数据
    features_path = "d:/Mathematical modeling/MCM_C/Q4_ver1/dance_competition_features.xlsx"
    processed_path = "d:/Mathematical modeling/MCM_C/Q4_ver1/dance_competition_final_processed.xlsx"
    fan_votes_path = "d:/Mathematical modeling/MCM_C/Q4_ver1/fan_vote_estimates.xlsx"

    df = model.load_data(features_path, processed_path, fan_votes_path)

    # 创建控制面板
    panel = ProducerControlPanel(model)

    # 显示可用预设
    panel.show_presets()

    # 选择一个赛季进行测试
    available_seasons = sorted(df['season'].unique())
    print(f"\n可用赛季: {available_seasons[:10]}...")  # 显示前10个

    test_season = 1  # 测试第1季

    print(f"\n{'='*70}")
    print(f"测试赛季: {test_season}")
    print(f"{'='*70}\n")

    # 应用平衡模式
    panel.apply_preset('平衡模式')

    # 模拟赛季
    results = model.simulate_season(test_season, total_weeks=6)

    # 可视化结果
    if results:
        visualize_results(results, save_path=f"d:/Mathematical modeling/MCM_C/Q4_ver1/season_{test_season}_results.png")

    # 对比不同模式
    print(f"\n{'='*70}")
    print("对比不同模式")
    print(f"{'='*70}\n")

    comparison_results = compare_methods(model, test_season)

    print("\n" + "="*70)
    print(" " * 20 + "模拟完成!")
    print("="*70)


if __name__ == "__main__":
    main()
