"""
舞蹈比赛双分支神经网络分析系统
=================================
构建双分支神经网络，同时预测评委评分和粉丝投票，并使用SHAP进行特征解释
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import shap

import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# 设置设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# 创建输出目录
OUTPUT_DIR = Path("outputs")
METRICS_DIR = OUTPUT_DIR / "metrics"
SHAP_DIR = OUTPUT_DIR / "shap_results"
CONTROVERSIAL_DIR = OUTPUT_DIR / "controversial_cases"
MODELS_DIR = Path("models")

for dir_path in [OUTPUT_DIR, METRICS_DIR, SHAP_DIR, CONTROVERSIAL_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 第一阶段：数据工程
# ============================================================================

class DataIntegrator:
    """数据加载与整合"""
    
    def __init__(self):
        self.data_dir = Path(".")
        
    def load_all_data(self):
        """加载所有必要的数据源"""
        print("=" * 80)
        print("步骤1：加载数据")
        print("=" * 80)
        
        # 1. 加载主数据集（包含评委评分和选手特征）
        main_df = pd.read_csv(self.data_dir / "dance_competition_final_processed.csv", encoding='gbk')
        print(f"✓ 加载主数据集: {main_df.shape}")
        
        # 2. 加载粉丝投票估计数据（来自Q1）
        fan_votes_df = pd.read_excel(self.data_dir / "fan_vote_estimates.xlsx")
        print(f"✓ 加载粉丝投票数据: {fan_votes_df.shape}")
        
        # 3. 加载Q2争议选手数据
        controversial_df = pd.read_excel(self.data_dir / "Q2_data" / "micro_analysis.xlsx")
        print(f"✓ 加载争议选手数据: {controversial_df.shape}")
        
        return main_df, fan_votes_df, controversial_df
    
    def integrate_datasets(self, main_df, fan_votes_df, controversial_df):
        """整合所有数据源"""
        print("\n" + "=" * 80)
        print("步骤2：整合数据")
        print("=" * 80)
        
        # 1. 首先处理主数据集
        df = main_df.copy()
        
        # 2. 从粉丝投票数据中选择需要的列并合并
        # 使用fan_votes作为目标变量（使用均值估计）
        fan_cols = ['season', 'week', 'celebrity_name', 'fan_votes']
        if 'fan_votes' in fan_votes_df.columns:
            fan_merge = fan_votes_df[fan_cols].copy()
        else:
            # 如果没有fan_votes，可能有fan_votes_mean
            fan_merge = fan_votes_df[['season', 'week', 'celebrity_name', 'fan_votes_mean']].copy()
            fan_merge.rename(columns={'fan_votes_mean': 'fan_votes'}, inplace=True)
        
        # 合并粉丝投票数据
        df = df.merge(fan_merge, 
                     on=['season', 'week', 'celebrity_name'],
                     how='left')
        print(f"✓ 合并粉丝投票数据后: {df.shape}")
        
        # 3. 添加争议选手标记
        controversial_names = set(controversial_df['player_name'].unique())
        df['is_controversial'] = df['celebrity_name'].apply(
            lambda x: 1 if x in controversial_names else 0
        )
        print(f"✓ 标记争议选手: {df['is_controversial'].sum()} 个争议选手记录")
        
        # 4. 添加投票机制信息
        # 根据赛季判断投票机制（赛季1-10为排名法，赛季11+为百分比法）
        df['voting_mechanism'] = df['season'].apply(
            lambda x: 'rank_based' if x <= 10 else 'percentage_based'
        )
        
        print(f"✓ 最终整合数据集: {df.shape}")
        print(f"✓ 包含列: {df.columns.tolist()}")
        
        return df


class DataCleaner:
    """数据清洗"""
    
    def clean_data(self, df):
        """执行数据清洗"""
        print("\n" + "=" * 80)
        print("步骤3：数据清洗")
        print("=" * 80)
        
        initial_rows = len(df)
        
        # 1. 目标变量完整性检查
        # 确保有评委评分（total_judge_score）和粉丝投票（fan_votes）
        print(f"评委评分缺失: {df['total_judge_score'].isna().sum()}")
        print(f"粉丝投票缺失: {df['fan_votes'].isna().sum()}")
        
        # 删除目标变量缺失的行
        df = df.dropna(subset=['total_judge_score', 'fan_votes'])
        print(f"✓ 删除目标变量缺失行: {initial_rows - len(df)} 行")
        
        # 2. 只保留正常比赛数据
        if 'in_competition' in df.columns:
            df = df[df['in_competition'] == True].copy()
            print(f"✓ 保留正常比赛数据: {len(df)} 行")
        
        # 3. 处理分类特征缺失值
        categorical_cols = ['celebrity_industry', 'ballroom_partner', 
                          'celebrity_homestate', 'celebrity_homecountry/region']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # 4. 处理数值特征缺失值
        numeric_cols = ['celebrity_age_during_season', 'relative_level']
        for col in numeric_cols:
            if col in df.columns and df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"✓ 填充 {col} 缺失值: 中位数 = {median_val:.2f}")
        
        # 5. 异常值处理（使用IQR方法）
        for col in ['total_judge_score', 'fan_votes']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"✓ {col} 异常值: {outliers} 个（保留但注意）")
        
        print(f"✓ 清洗后数据集: {df.shape}")
        
        return df


class FeatureEngineer:
    """特征工程"""
    
    def __init__(self):
        self.categorical_cols = []
        self.numeric_cols = []
        self.scaler = None
        self.encoders = {}
        
    def engineer_features(self, df):
        """执行特征工程"""
        print("\n" + "=" * 80)
        print("步骤4：特征工程")
        print("=" * 80)
        
        df = df.copy()
        
        # 1. 定义特征列
        # 数值特征
        self.numeric_cols = [
            'celebrity_age_during_season',
            'relative_level',
            'week',
            'season',
            'relative_score_to_week_avg'
        ]
        
        # 分类特征
        self.categorical_cols = [
            'celebrity_industry',
            'ballroom_partner',
            'season_stage',
            'celebrity_age_during_season_bin',
            'voting_mechanism',
            'is_controversial'
        ]
        
        # 确保所有特征列存在
        available_numeric = [col for col in self.numeric_cols if col in df.columns]
        available_categorical = [col for col in self.categorical_cols if col in df.columns]
        
        self.numeric_cols = available_numeric
        self.categorical_cols = available_categorical
        
        print(f"✓ 数值特征 ({len(self.numeric_cols)}): {self.numeric_cols}")
        print(f"✓ 分类特征 ({len(self.categorical_cols)}): {self.categorical_cols}")
        
        # 2. 创建特征交叉（可选）
        if 'celebrity_industry' in df.columns and 'is_controversial' in df.columns:
            df['industry_controversial'] = (
                df['celebrity_industry'].astype(str) + '_' + 
                df['is_controversial'].astype(str)
            )
            self.categorical_cols.append('industry_controversial')
            print("✓ 创建特征交叉: industry_controversial")
        
        # 3. 准备目标变量
        df['y_judge'] = df['total_judge_score']
        df['y_fan'] = df['fan_votes']
        
        print(f"✓ 目标变量统计:")
        print(f"  - 评委评分: 均值={df['y_judge'].mean():.2f}, 标准差={df['y_judge'].std():.2f}")
        print(f"  - 粉丝投票: 均值={df['y_fan'].mean():.2f}, 标准差={df['y_fan'].std():.2f}")
        
        return df
    
    def prepare_features_for_training(self, df_train, df_val, df_test):
        """为训练准备特征（编码和标准化）"""
        print("\n" + "=" * 80)
        print("步骤5：特征编码与标准化")
        print("=" * 80)
        
        # 1. 编码分类特征
        encoded_train_dfs = []
        encoded_val_dfs = []
        encoded_test_dfs = []
        
        for col in self.categorical_cols:
            if col in df_train.columns:
                # 使用LabelEncoder（对于树模型）或OneHot（对于神经网络）
                # 这里使用LabelEncoder简化
                le = LabelEncoder()
                
                # Fit on training data
                le.fit(df_train[col].astype(str))
                self.encoders[col] = le
                
                # Transform all sets
                df_train[f'{col}_encoded'] = le.transform(df_train[col].astype(str))
                
                # Handle unseen categories in val/test
                df_val[f'{col}_encoded'] = df_val[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                df_test[f'{col}_encoded'] = df_test[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                
        print(f"✓ 编码 {len(self.categorical_cols)} 个分类特征")
        
        # 2. 标准化数值特征
        self.scaler = StandardScaler()
        
        # Fit on training data
        df_train[self.numeric_cols] = self.scaler.fit_transform(
            df_train[self.numeric_cols]
        )
        
        # Transform val and test
        df_val[self.numeric_cols] = self.scaler.transform(
            df_val[self.numeric_cols]
        )
        df_test[self.numeric_cols] = self.scaler.transform(
            df_test[self.numeric_cols]
        )
        
        print(f"✓ 标准化 {len(self.numeric_cols)} 个数值特征")
        
        # 3. 收集所有特征列
        encoded_categorical = [f'{col}_encoded' for col in self.categorical_cols]
        all_feature_cols = self.numeric_cols + encoded_categorical
        
        print(f"✓ 总特征数: {len(all_feature_cols)}")
        
        return df_train, df_val, df_test, all_feature_cols


class DataSplitter:
    """数据集划分"""
    
    def split_by_celebrity(self, df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """按选手划分数据集，防止数据泄露"""
        print("\n" + "=" * 80)
        print("步骤6：数据集划分")
        print("=" * 80)
        
        # 获取所有唯一选手
        celebrities = df['celebrity_name'].unique()
        print(f"✓ 总选手数: {len(celebrities)}")
        
        # 随机划分选手
        np.random.shuffle(celebrities)
        
        n_train = int(len(celebrities) * train_ratio)
        n_val = int(len(celebrities) * val_ratio)
        
        train_celebrities = celebrities[:n_train]
        val_celebrities = celebrities[n_train:n_train + n_val]
        test_celebrities = celebrities[n_train + n_val:]
        
        # 根据选手划分数据
        df_train = df[df['celebrity_name'].isin(train_celebrities)].copy()
        df_val = df[df['celebrity_name'].isin(val_celebrities)].copy()
        df_test = df[df['celebrity_name'].isin(test_celebrities)].copy()
        
        print(f"✓ 训练集: {len(df_train)} 行, {len(train_celebrities)} 个选手")
        print(f"✓ 验证集: {len(df_val)} 行, {len(val_celebrities)} 个选手")
        print(f"✓ 测试集: {len(df_test)} 行, {len(test_celebrities)} 个选手")
        
        return df_train, df_val, df_test


# ============================================================================
# 第二阶段：模型构建与训练
# ============================================================================

class DualBranchNet(nn.Module):
    """双分支神经网络"""
    
    def __init__(self, input_dim, shared_dims=[128, 64], branch_dims=[32], dropout=0.3):
        super(DualBranchNet, self).__init__()
        
        # 共享特征提取层
        shared_layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # 评委评分分支
        judge_layers = []
        prev_dim = shared_dims[-1]
        for dim in branch_dims:
            judge_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        judge_layers.append(nn.Linear(prev_dim, 1))
        self.judge_branch = nn.Sequential(*judge_layers)
        
        # 粉丝投票分支
        fan_layers = []
        prev_dim = shared_dims[-1]
        for dim in branch_dims:
            fan_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        fan_layers.append(nn.Linear(prev_dim, 1))
        self.fan_branch = nn.Sequential(*fan_layers)
    
    def forward(self, x):
        """前向传播"""
        shared_features = self.shared_layers(x)
        judge_output = self.judge_branch(shared_features)
        fan_output = self.fan_branch(shared_features)
        return judge_output, fan_output


class WeightedMultiTaskLoss(nn.Module):
    """加权多任务损失函数"""
    
    def __init__(self, alpha=1.0, beta=1.0):
        super(WeightedMultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, judge_pred, judge_true, fan_pred, fan_true):
        """计算加权损失"""
        loss_judge = self.mse(judge_pred, judge_true)
        loss_fan = self.mse(fan_pred, fan_true)
        total_loss = self.alpha * loss_judge + self.beta * loss_fan
        return total_loss, loss_judge, loss_fan


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, train_loader, val_loader, device=DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def setup_training(self, lr=0.001, alpha=1.0, beta=1.0):
        """设置训练组件"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = WeightedMultiTaskLoss(alpha=alpha, beta=beta)
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_judge_loss = 0
        total_fan_loss = 0
        
        for X_batch, y_judge_batch, y_fan_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_judge_batch = y_judge_batch.to(self.device)
            y_fan_batch = y_fan_batch.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            judge_pred, fan_pred = self.model(X_batch)
            
            # 计算损失
            loss, loss_judge, loss_fan = self.criterion(
                judge_pred, y_judge_batch,
                fan_pred, y_fan_batch
            )
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_judge_loss += loss_judge.item()
            total_fan_loss += loss_fan.item()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_judge_loss = total_judge_loss / len(self.train_loader)
        avg_fan_loss = total_fan_loss / len(self.train_loader)
        
        return avg_loss, avg_judge_loss, avg_fan_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_judge_loss = 0
        total_fan_loss = 0
        
        with torch.no_grad():
            for X_batch, y_judge_batch, y_fan_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_judge_batch = y_judge_batch.to(self.device)
                y_fan_batch = y_fan_batch.to(self.device)
                
                judge_pred, fan_pred = self.model(X_batch)
                loss, loss_judge, loss_fan = self.criterion(
                    judge_pred, y_judge_batch,
                    fan_pred, y_fan_batch
                )
                
                total_loss += loss.item()
                total_judge_loss += loss_judge.item()
                total_fan_loss += loss_fan.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_judge_loss = total_judge_loss / len(self.val_loader)
        avg_fan_loss = total_fan_loss / len(self.val_loader)
        
        return avg_loss, avg_judge_loss, avg_fan_loss
    
    def train(self, epochs=100, patience=10):
        """完整训练流程"""
        print("\n" + "=" * 80)
        print("开始训练双分支神经网络")
        print("=" * 80)
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_judge_loss, train_fan_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_judge_loss, val_fan_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  训练损失: {train_loss:.4f} (评委: {train_judge_loss:.4f}, 粉丝: {train_fan_loss:.4f})")
                print(f"  验证损失: {val_loss:.4f} (评委: {val_judge_loss:.4f}, 粉丝: {val_fan_loss:.4f})")
                print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), MODELS_DIR / 'best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n早停触发! 验证损失连续{patience}个epoch未改善")
                    break
        
        print(f"\n训练完成! 最佳验证损失: {self.best_val_loss:.4f}")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(MODELS_DIR / 'best_model.pth'))
        
        return self.model


def create_data_loaders(df_train, df_val, df_test, feature_cols, batch_size=64):
    """创建PyTorch数据加载器"""

    # 标准化目标变量
    judge_scaler = StandardScaler()
    fan_scaler = StandardScaler()

    # 准备特征和目标
    X_train = torch.FloatTensor(df_train[feature_cols].values)
    y_judge_train = torch.FloatTensor(judge_scaler.fit_transform(df_train['y_judge'].values.reshape(-1, 1)))
    y_fan_train = torch.FloatTensor(fan_scaler.fit_transform(df_train['y_fan'].values.reshape(-1, 1)))

    X_val = torch.FloatTensor(df_val[feature_cols].values)
    y_judge_val = torch.FloatTensor(judge_scaler.transform(df_val['y_judge'].values.reshape(-1, 1)))
    y_fan_val = torch.FloatTensor(fan_scaler.transform(df_val['y_fan'].values.reshape(-1, 1)))

    X_test = torch.FloatTensor(df_test[feature_cols].values)
    y_judge_test = torch.FloatTensor(judge_scaler.transform(df_test['y_judge'].values.reshape(-1, 1)))
    y_fan_test = torch.FloatTensor(fan_scaler.transform(df_test['y_fan'].values.reshape(-1, 1)))

    # 创建数据集
    train_dataset = TensorDataset(X_train, y_judge_train, y_fan_train)
    val_dataset = TensorDataset(X_val, y_judge_val, y_fan_val)
    test_dataset = TensorDataset(X_test, y_judge_test, y_fan_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"✓ 创建数据加载器: batch_size={batch_size}")
    print(f"  - 训练批次: {len(train_loader)}")
    print(f"  - 验证批次: {len(val_loader)}")
    print(f"  - 测试批次: {len(test_loader)}")

    return train_loader, val_loader, test_loader, (X_test, y_judge_test, y_fan_test), judge_scaler, fan_scaler


# ============================================================================
# 第三阶段：模型评估与对比
# ============================================================================

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device=DEVICE):
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader, df_test, task_name="测试集"):
        """评估模型性能"""
        print("\n" + "=" * 80)
        print(f"评估模型: {task_name}")
        print("=" * 80)
        
        self.model.eval()
        
        all_judge_preds = []
        all_judge_true = []
        all_fan_preds = []
        all_fan_true = []
        
        with torch.no_grad():
            for X_batch, y_judge_batch, y_fan_batch in test_loader:
                X_batch = X_batch.to(self.device)
                
                judge_pred, fan_pred = self.model(X_batch)
                
                all_judge_preds.extend(judge_pred.cpu().numpy())
                all_judge_true.extend(y_judge_batch.numpy())
                all_fan_preds.extend(fan_pred.cpu().numpy())
                all_fan_true.extend(y_fan_batch.numpy())
        
        all_judge_preds = np.array(all_judge_preds).flatten()
        all_judge_true = np.array(all_judge_true).flatten()
        all_fan_preds = np.array(all_fan_preds).flatten()
        all_fan_true = np.array(all_fan_true).flatten()
        
        # 计算评估指标
        metrics = self._calculate_metrics(
            all_judge_preds, all_judge_true,
            all_fan_preds, all_fan_true
        )
        
        # 打印结果
        self._print_metrics(metrics)
        
        # 保存结果
        self._save_predictions(df_test, all_judge_preds, all_fan_preds)
        
        return metrics, (all_judge_preds, all_judge_true, all_fan_preds, all_fan_true)
    
    def _calculate_metrics(self, judge_pred, judge_true, fan_pred, fan_true):
        """计算评估指标"""
        metrics = {
            'judge': {
                'mse': mean_squared_error(judge_true, judge_pred),
                'rmse': np.sqrt(mean_squared_error(judge_true, judge_pred)),
                'mae': mean_absolute_error(judge_true, judge_pred),
                'r2': r2_score(judge_true, judge_pred),
                'mape': np.mean(np.abs((judge_true - judge_pred) / (judge_true + 1e-8))) * 100
            },
            'fan': {
                'mse': mean_squared_error(fan_true, fan_pred),
                'rmse': np.sqrt(mean_squared_error(fan_true, fan_pred)),
                'mae': mean_absolute_error(fan_true, fan_pred),
                'r2': r2_score(fan_true, fan_pred),
                'mape': np.mean(np.abs((fan_true - fan_pred) / (fan_true + 1e-8))) * 100
            }
        }
        return metrics
    
    def _print_metrics(self, metrics):
        """打印评估指标"""
        print("\n评委评分预测:")
        print(f"  MSE:  {metrics['judge']['mse']:.4f}")
        print(f"  RMSE: {metrics['judge']['rmse']:.4f}")
        print(f"  MAE:  {metrics['judge']['mae']:.4f}")
        print(f"  R²:   {metrics['judge']['r2']:.4f}")
        print(f"  MAPE: {metrics['judge']['mape']:.2f}%")
        
        print("\n粉丝投票预测:")
        print(f"  MSE:  {metrics['fan']['mse']:.4f}")
        print(f"  RMSE: {metrics['fan']['rmse']:.4f}")
        print(f"  MAE:  {metrics['fan']['mae']:.4f}")
        print(f"  R²:   {metrics['fan']['r2']:.4f}")
        print(f"  MAPE: {metrics['fan']['mape']:.2f}%")
    
    def _save_predictions(self, df_test, judge_preds, fan_preds):
        """保存预测结果"""
        results_df = df_test.copy()
        results_df['judge_pred'] = judge_preds
        results_df['fan_pred'] = fan_preds
        results_df.to_csv(METRICS_DIR / 'predictions.csv', index=False)
        print(f"\n✓ 保存预测结果到: {METRICS_DIR / 'predictions.csv'}")


class BaselineComparator:
    """基准模型对比"""
    
    def train_baseline_models(self, df_train, df_val, feature_cols):
        """训练XGBoost基准模型"""
        print("\n" + "=" * 80)
        print("训练XGBoost基准模型")
        print("=" * 80)
        
        X_train = df_train[feature_cols].values
        X_val = df_val[feature_cols].values
        
        # 评委评分模型
        print("\n训练评委评分预测模型...")
        y_judge_train = df_train['y_judge'].values
        y_judge_val = df_val['y_judge'].values
        
        model_judge = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=10
        )
        model_judge.fit(
            X_train, y_judge_train,
            eval_set=[(X_val, y_judge_val)],
            verbose=False
        )
        print(f"✓ 评委评分模型训练完成")
        
        # 粉丝投票模型
        print("\n训练粉丝投票预测模型...")
        y_fan_train = df_train['y_fan'].values
        y_fan_val = df_val['y_fan'].values
        
        model_fan = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=10
        )
        model_fan.fit(
            X_train, y_fan_train,
            eval_set=[(X_val, y_fan_val)],
            verbose=False
        )
        print(f"✓ 粉丝投票模型训练完成")
        
        return model_judge, model_fan
    
    def compare_with_baseline(self, nn_metrics, baseline_models, df_test, feature_cols):
        """对比神经网络和基准模型"""
        print("\n" + "=" * 80)
        print("基准模型对比")
        print("=" * 80)
        
        model_judge, model_fan = baseline_models
        
        X_test = df_test[feature_cols].values
        y_judge_test = df_test['y_judge'].values
        y_fan_test = df_test['y_fan'].values
        
        # 预测
        judge_pred = model_judge.predict(X_test)
        fan_pred = model_fan.predict(X_test)
        
        # 计算指标
        baseline_metrics = {
            'judge': {
                'mse': mean_squared_error(y_judge_test, judge_pred),
                'rmse': np.sqrt(mean_squared_error(y_judge_test, judge_pred)),
                'mae': mean_absolute_error(y_judge_test, judge_pred),
                'r2': r2_score(y_judge_test, judge_pred),
            },
            'fan': {
                'mse': mean_squared_error(y_fan_test, fan_pred),
                'rmse': np.sqrt(mean_squared_error(y_fan_test, fan_pred)),
                'mae': mean_absolute_error(y_fan_test, fan_pred),
                'r2': r2_score(y_fan_test, fan_pred),
            }
        }
        
        # 打印对比
        print("\n模型对比:")
        print("-" * 60)
        print(f"{'指标':<15} {'神经网络(评委)':<20} {'XGBoost(评委)':<20}")
        print("-" * 60)
        print(f"{'R²':<15} {nn_metrics['judge']['r2']:<20.4f} {baseline_metrics['judge']['r2']:<20.4f}")
        print(f"{'RMSE':<15} {nn_metrics['judge']['rmse']:<20.4f} {baseline_metrics['judge']['rmse']:<20.4f}")
        print(f"{'MAE':<15} {nn_metrics['judge']['mae']:<20.4f} {baseline_metrics['judge']['mae']:<20.4f}")
        
        print("\n")
        print("-" * 60)
        print(f"{'指标':<15} {'神经网络(粉丝)':<20} {'XGBoost(粉丝)':<20}")
        print("-" * 60)
        print(f"{'R²':<15} {nn_metrics['fan']['r2']:<20.4f} {baseline_metrics['fan']['r2']:<20.4f}")
        print(f"{'RMSE':<15} {nn_metrics['fan']['rmse']:<20.4f} {baseline_metrics['fan']['rmse']:<20.4f}")
        print(f"{'MAE':<15} {nn_metrics['fan']['mae']:<20.4f} {baseline_metrics['fan']['mae']:<20.4f}")
        
        # 保存对比结果
        comparison_df = pd.DataFrame({
            'Model': ['DualBranchNet', 'XGBoost'],
            'Judge_R2': [nn_metrics['judge']['r2'], baseline_metrics['judge']['r2']],
            'Judge_RMSE': [nn_metrics['judge']['rmse'], baseline_metrics['judge']['rmse']],
            'Fan_R2': [nn_metrics['fan']['r2'], baseline_metrics['fan']['r2']],
            'Fan_RMSE': [nn_metrics['fan']['rmse'], baseline_metrics['fan']['rmse']]
        })
        comparison_df.to_csv(METRICS_DIR / 'baseline_comparison.csv', index=False)
        print(f"\n✓ 保存对比结果到: {METRICS_DIR / 'baseline_comparison.csv'}")
        
        return baseline_metrics


# ============================================================================
# 第四阶段：模型解释与可视化
# ============================================================================

class SHAPAnalyzer:
    """SHAP分析器"""
    
    def __init__(self, model, device=DEVICE):
        self.model = model
        self.device = device

    def analyze_global_shap(self, X_test_tensor, feature_names, n_background=100):
        """全局SHAP分析 - 修改版本，使用 KernelExplainer"""
        print("\n" + "=" * 80)
        print("SHAP全局解释分析 (使用KernelExplainer)")
        print("=" * 80)

        # 将数据移动到CPU并转换为numpy
        X_test_np = X_test_tensor.cpu().numpy()

        # 准备背景数据
        if len(X_test_np) > n_background:
            indices = np.random.choice(len(X_test_np), n_background, replace=False)
            background = X_test_np[indices]
        else:
            background = X_test_np

        # 定义模型包装器
        def model_judge_wrapper(x):
            """评委评分预测包装器"""
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(self.device)
                judge_out, _ = self.model(x_tensor)
                return judge_out.cpu().numpy()

        def model_fan_wrapper(x):
            """粉丝投票预测包装器"""
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(self.device)
                _, fan_out = self.model(x_tensor)
                return fan_out.cpu().numpy()

        # 创建KernelExplainer
        print(f"✓ 创建KernelExplainer (背景样本数: {len(background)})")

        # 评委评分分支
        print("计算评委评分SHAP值...")
        explainer_judge = shap.KernelExplainer(
            model_judge_wrapper,
            background,
            link="identity"
        )
        shap_values_judge = explainer_judge.shap_values(X_test_np[:50], nsamples=100)  # 只计算前50个样本

        # 粉丝投票分支
        print("计算粉丝投票SHAP值...")
        explainer_fan = shap.KernelExplainer(
            model_fan_wrapper,
            background,
            link="identity"
        )
        shap_values_fan = explainer_fan.shap_values(X_test_np[:50], nsamples=100)  # 只计算前50个样本

        # 调试信息
        print(f"✓ SHAP计算完成")
        print(f"  - 评委SHAP形状: {np.shape(shap_values_judge)}")
        print(f"  - 粉丝SHAP形状: {np.shape(shap_values_fan)}")

        # 确保SHAP值是二维数组
        if len(np.shape(shap_values_judge)) > 2:
            shap_values_judge = shap_values_judge.reshape(shap_values_judge.shape[0], -1)
        if len(np.shape(shap_values_fan)) > 2:
            shap_values_fan = shap_values_fan.reshape(shap_values_fan.shape[0], -1)

        print(f"  - 处理后的评委SHAP形状: {np.shape(shap_values_judge)}")
        print(f"  - 处理后的粉丝SHAP形状: {np.shape(shap_values_fan)}")

        # 可视化
        self._visualize_shap(shap_values_judge, shap_values_fan,
                             X_test_np[:50], feature_names)

        # 生成特征影响力报告
        self._generate_feature_impact_report(shap_values_judge, shap_values_fan, feature_names)

        return shap_values_judge, shap_values_fan

    def _visualize_shap(self, shap_judge, shap_fan, test_data, feature_names):
        """可视化SHAP结果"""
        print("\n生成SHAP可视化...")

        # 确保SHAP值是二维数组
        if len(shap_judge.shape) > 2:
            shap_judge = shap_judge.reshape(shap_judge.shape[0], -1)
        if len(shap_fan.shape) > 2:
            shap_fan = shap_fan.reshape(shap_fan.shape[0], -1)

        # 确保特征名称与SHAP值维度匹配
        if shap_judge.shape[1] != len(feature_names):
            # 调整特征名称
            feature_names = feature_names[:shap_judge.shape[1]]

        # 评委评分SHAP摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_judge, test_data[:, :shap_judge.shape[1]],
                          feature_names=feature_names, show=False)
        plt.title('Impact of Features on Review Scores (SHAP)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(SHAP_DIR / 'global_shap_judge.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存评委评分SHAP图: {SHAP_DIR / 'global_shap_judge.png'}")

        # 粉丝投票SHAP摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_fan, test_data[:, :shap_fan.shape[1]],
                          feature_names=feature_names, show=False)
        plt.title('Impact of Features on Fan Voting (SHAP)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(SHAP_DIR / 'global_shap_fan.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存粉丝投票SHAP图: {SHAP_DIR / 'global_shap_fan.png'}")

        # 特征重要性对比
        self._plot_feature_importance_comparison(shap_judge, shap_fan, feature_names)

    def _plot_feature_importance_comparison(self, shap_judge, shap_fan, feature_names):
        """绘制特征重要性对比图"""
        # 确保SHAP值是一维或二维数组
        if len(shap_judge.shape) > 2:
            shap_judge = shap_judge.reshape(shap_judge.shape[0], -1)
        if len(shap_fan.shape) > 2:
            shap_fan = shap_fan.reshape(shap_fan.shape[0], -1)

        # 计算平均绝对SHAP值，确保结果是一维数组
        judge_importance = np.abs(shap_judge).mean(axis=0)
        fan_importance = np.abs(shap_fan).mean(axis=0)

        # 如果仍然不是一维，进行展平
        if len(judge_importance.shape) > 1:
            judge_importance = judge_importance.flatten()
        if len(fan_importance.shape) > 1:
            fan_importance = fan_importance.flatten()

        # 确保特征名称与重要性数组长度匹配
        min_len = min(len(feature_names), len(judge_importance), len(fan_importance))
        feature_names = feature_names[:min_len]
        judge_importance = judge_importance[:min_len]
        fan_importance = fan_importance[:min_len]

        # 创建DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Judge_Importance': judge_importance,
            'Fan_Importance': fan_importance
        })

        # 按总重要性排序
        importance_df['Total'] = importance_df['Judge_Importance'] + importance_df['Fan_Importance']
        importance_df = importance_df.sort_values('Total', ascending=True).tail(15)

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(importance_df))
        width = 0.35

        ax.barh(y_pos - width / 2, importance_df['Judge_Importance'],
                width, label='Judge Score', alpha=0.8, color='#1f77b4')
        ax.barh(y_pos + width / 2, importance_df['Fan_Importance'],
                width, label='Fan Vote', alpha=0.8, color='#ff7f0e')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['Feature'])
        ax.set_xlabel('Average|SHAP Value|', fontsize=12)
        ax.set_title('Feature Importance Comparison: Judges vs. Fans', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(SHAP_DIR / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存特征重要性对比图: {SHAP_DIR / 'feature_importance_comparison.png'}")

    # 修改 _generate_feature_impact_report 方法
    def _generate_feature_impact_report(self, shap_judge, shap_fan, feature_names):
        """生成特征影响力报告"""
        # 确保SHAP值是二维数组
        if len(shap_judge.shape) > 2:
            shap_judge = shap_judge.reshape(shap_judge.shape[0], -1)
        if len(shap_fan.shape) > 2:
            shap_fan = shap_fan.reshape(shap_fan.shape[0], -1)

        # 计算平均值
        judge_mean_shap = shap_judge.mean(axis=0)
        fan_mean_shap = shap_fan.mean(axis=0)

        # 计算重要性
        judge_importance = np.abs(shap_judge).mean(axis=0)
        fan_importance = np.abs(shap_fan).mean(axis=0)

        # 确保所有数组都是一维的
        if len(judge_mean_shap.shape) > 1:
            judge_mean_shap = judge_mean_shap.flatten()
        if len(fan_mean_shap.shape) > 1:
            fan_mean_shap = fan_mean_shap.flatten()
        if len(judge_importance.shape) > 1:
            judge_importance = judge_importance.flatten()
        if len(fan_importance.shape) > 1:
            fan_importance = fan_importance.flatten()

        # 确保数组长度匹配特征数量
        min_len = min(len(feature_names), len(judge_mean_shap), len(fan_mean_shap),
                      len(judge_importance), len(fan_importance))

        feature_names = feature_names[:min_len]
        judge_mean_shap = judge_mean_shap[:min_len]
        fan_mean_shap = fan_mean_shap[:min_len]
        judge_importance = judge_importance[:min_len]
        fan_importance = fan_importance[:min_len]

        # 归一化重要性
        judge_importance_norm = judge_importance / (judge_importance.sum() + 1e-8)
        fan_importance_norm = fan_importance / (fan_importance.sum() + 1e-8)

        # 确定影响类型
        impact_types = []
        for i in range(min_len):
            j_mean = judge_mean_shap[i]
            f_mean = fan_mean_shap[i]
            j_imp = judge_importance[i]
            f_imp = fan_importance[i]

            if j_imp < 0.01 and f_imp < 0.01:
                impact_types.append('no_impact')
            elif abs(j_imp - f_imp) / max(j_imp, f_imp, 1e-8) < 0.3:
                if j_mean > 0 and f_mean > 0:
                    impact_types.append('synergistic_positive')
                elif j_mean < 0 and f_mean < 0:
                    impact_types.append('synergistic_negative')
                else:
                    impact_types.append('synergistic_mixed')
            elif j_mean * f_mean < 0:
                impact_types.append('opposing')
            elif j_imp > f_imp * 1.5:
                impact_types.append('judge_favored')
            elif f_imp > j_imp * 1.5:
                impact_types.append('fan_favored')
            else:
                impact_types.append('balanced')

        # 创建报告DataFrame
        report_df = pd.DataFrame({
            'feature': feature_names,
            'judge_mean_shap': judge_mean_shap,
            'fan_mean_shap': fan_mean_shap,
            'judge_importance': judge_importance_norm,
            'fan_importance': fan_importance_norm,
            'impact_type': impact_types
        })

        report_df = report_df.sort_values('judge_importance', ascending=False)
        report_df.to_csv(SHAP_DIR / 'feature_impact_report.csv', index=False)
        print(f"✓ 保存特征影响力报告: {SHAP_DIR / 'feature_impact_report.csv'}")

        # 打印摘要
        print("\n特征影响类型统计:")
        print(report_df['impact_type'].value_counts())


class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_training_history(train_losses, val_losses):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training loss', linewidth=2)
        plt.plot(val_losses, label='Verification loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Process Loss Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(METRICS_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存训练历史图: {METRICS_DIR / 'training_history.png'}")
    
    @staticmethod
    def plot_predictions(judge_pred, judge_true, fan_pred, fan_true):
        """绘制预测值vs真实值"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 评委评分
        axes[0].scatter(judge_true, judge_pred, alpha=0.5, s=20)
        axes[0].plot([judge_true.min(), judge_true.max()], 
                    [judge_true.min(), judge_true.max()], 
                    'r--', lw=2, label='Ideal Prediction Line')
        axes[0].set_xlabel('Actual Judge Scores', fontsize=12)
        axes[0].set_ylabel('Predicted Judge Scores', fontsize=12)
        axes[0].set_title('Predicting Judge Scores', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 粉丝投票
        axes[1].scatter(fan_true, fan_pred, alpha=0.5, s=20, color='orange')
        axes[1].plot([fan_true.min(), fan_true.max()], 
                    [fan_true.min(), fan_true.max()], 
                    'r--', lw=2, label='deal Prediction Line')
        axes[1].set_xlabel('Real Fan Votes', fontsize=12)
        axes[1].set_ylabel('Predicted Fan Votes', fontsize=12)
        axes[1].set_title('Predicting Fan Votes', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(METRICS_DIR / 'predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存预测对比图: {METRICS_DIR / 'predictions_vs_actual.png'}")
    
    @staticmethod
    def plot_residuals(judge_pred, judge_true, fan_pred, fan_true):
        """绘制残差图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        judge_residuals = judge_true - judge_pred
        fan_residuals = fan_true - fan_pred
        
        # 评委评分残差
        axes[0].scatter(judge_pred, judge_residuals, alpha=0.5, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Forecast of Judge Scores', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title('Distribution of Residuals in Judge Scoring', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # 粉丝投票残差
        axes[1].scatter(fan_pred, fan_residuals, alpha=0.5, s=20, color='orange')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Forecast of Fan Votes', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title('Residual Distribution of Fan Voting', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(METRICS_DIR / 'residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存残差图: {METRICS_DIR / 'residuals.png'}")


# ============================================================================
# 主执行流程
# ============================================================================

def main():
    """主执行函数"""
    print("\n" + "=" * 80)
    print("舞蹈比赛双分支神经网络分析系统")
    print("=" * 80)
    
    # ========================================================================
    # 阶段1: 数据工程
    # ========================================================================
    
    # 步骤1-2: 数据加载与整合
    integrator = DataIntegrator()
    main_df, fan_votes_df, controversial_df = integrator.load_all_data()
    df = integrator.integrate_datasets(main_df, fan_votes_df, controversial_df)
    
    # 步骤3: 数据清洗
    cleaner = DataCleaner()
    df = cleaner.clean_data(df)
    
    # 步骤4: 特征工程
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    
    # 步骤5: 数据集划分
    splitter = DataSplitter()
    df_train, df_val, df_test = splitter.split_by_celebrity(df)
    
    # 步骤6: 特征编码与标准化
    df_train, df_val, df_test, feature_cols = engineer.prepare_features_for_training(
        df_train, df_val, df_test
    )
    
    # 步骤7: 创建数据加载器
    train_loader, val_loader, test_loader, test_tensors, judge_scaler, fan_scaler = create_data_loaders(
        df_train, df_val, df_test, feature_cols, batch_size=64
    )
    
    # ========================================================================
    # 阶段2: 模型构建与训练
    # ========================================================================

    print("\n" + "=" * 80)
    print("构建双分支神经网络")
    print("=" * 80)

    input_dim = len(feature_cols)
    model = DualBranchNet(
        input_dim=input_dim,
        shared_dims=[128, 64],
        branch_dims=[32],
        dropout=0.3
    )
    print(f"✓ 模型构建完成")
    print(f"  - 输入维度: {input_dim}")
    print(f"  - 总参数量: {sum(p.numel() for p in model.parameters())}")

    # 步骤9: 训练模型
    trainer = Trainer(model, train_loader, val_loader)
    trainer.setup_training(lr=0.001, alpha=1.0, beta=1.0)
    trained_model = trainer.train(epochs=100, patience=15)

    # 绘制训练历史
    Visualizer.plot_training_history(trainer.train_losses, trainer.val_losses)

    # ========================================================================
    # 阶段3: 模型评估与对比
    # ========================================================================

    # 步骤10: 评估双分支神经网络
    evaluator = ModelEvaluator(trained_model)
    nn_metrics, predictions = evaluator.evaluate(test_loader, df_test)
    judge_pred, judge_true, fan_pred, fan_true = predictions

    # 绘制预测结果
    Visualizer.plot_predictions(judge_pred, judge_true, fan_pred, fan_true)
    Visualizer.plot_residuals(judge_pred, judge_true, fan_pred, fan_true)

    # 步骤11: 训练并对比基准模型
    comparator = BaselineComparator()
    baseline_models = comparator.train_baseline_models(df_train, df_val, feature_cols)
    baseline_metrics = comparator.compare_with_baseline(
        nn_metrics, baseline_models, df_test, feature_cols
    )

    # ========================================================================
    # 阶段4: 模型解释与业务洞见
    # ========================================================================

    # 步骤12: SHAP全局解释
    # 注意：这里需要解包 test_tensors
    X_test_tensor, _, _ = test_tensors
    shap_analyzer = SHAPAnalyzer(trained_model)
    shap_judge, shap_fan = shap_analyzer.analyze_global_shap(
        X_test_tensor, feature_cols, n_background=100
    )
    
    # ========================================================================
    # 总结与输出
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
    print(f"\n所有输出文件已保存到:")
    print(f"  - 模型文件: {MODELS_DIR}")
    print(f"  - 评估指标: {METRICS_DIR}")
    print(f"  - SHAP分析: {SHAP_DIR}")
    print(f"  - 争议选手: {CONTROVERSIAL_DIR}")
    
    print("\n关键发现:")
    print(f"  1. 双分支神经网络在评委评分任务上的R² = {nn_metrics['judge']['r2']:.4f}")
    print(f"  2. 双分支神经网络在粉丝投票任务上的R² = {nn_metrics['fan']['r2']:.4f}")
    
    if nn_metrics['judge']['r2'] > 0.7 and nn_metrics['fan']['r2'] > 0.7:
        print("\n✓ 成功! 模型在两个任务上的R²分数都超过0.7")
    else:
        print("\n⚠ 注意: 模型性能未达到R²>0.7的目标,可能需要:")
        print("  - 调整超参数")
        print("  - 增加更多特征")
        print("  - 收集更多数据")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
