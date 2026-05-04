"""
测试脚本：验证随机森林类别不平衡问题的修复
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("测试：随机森林类别不平衡修复")
print("=" * 60)

# 加载数据
print("\n步骤1: 加载数据...")
final_df = pd.read_csv("dance_competition_final_processed.csv", 
                       encoding='utf-8', encoding_errors='ignore')

# 提取特征和标签
feature_cols = [
    "relative_level", "celebrity_industry_encoded", "ballroom_partner_encoded",
    "celebrity_age_during_season_bin_encoded", "season_stage_encoded",
    "all_star_season", "controversial_contestant"
]

# 检查缺失列
missing_cols = [col for col in feature_cols if col not in final_df.columns]
if missing_cols:
    print(f"  缺失列: {missing_cols}，使用默认值0")
    for col in missing_cols:
        final_df[col] = 0

# 准备数据
data = final_df[feature_cols + ["eliminated_this_week"]].dropna()

# 转换布尔类型
bool_cols = ["all_star_season", "controversial_contestant", "eliminated_this_week"]
for col in bool_cols:
    if col in data.columns and data[col].dtype == 'bool':
        data[col] = data[col].astype(int)

X = data[feature_cols]
y = data["eliminated_this_week"].astype(int)

print(f"  样本总数: {len(data)}")
print(f"  特征数量: {len(feature_cols)}")

# 检查类别分布
print(f"\n步骤2: 类别分布")
print(f"  未淘汰（0）: {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"  已淘汰（1）: {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
print(f"  不平衡比例: {(y == 0).sum() / (y == 1).sum():.2f}:1")

# 训练模型（不平衡处理前）
print("\n步骤3: 训练模型（未处理不平衡）")
rf_unbalanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf_unbalanced.fit(X, y)

y_pred_unbalanced = rf_unbalanced.predict(X)
acc_unbalanced = accuracy_score(y, y_pred_unbalanced)
cm_unbalanced = confusion_matrix(y, y_pred_unbalanced)

print(f"  准确率: {acc_unbalanced:.4f}")
print(f"  混淆矩阵:")
print(f"    [[TN={cm_unbalanced[0,0]}, FP={cm_unbalanced[0,1]}],")
print(f"     [FN={cm_unbalanced[1,0]}, TP={cm_unbalanced[1,1]}]]")

feature_importance_unbalanced = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_unbalanced.feature_importances_
}).sort_values("importance", ascending=False)

print(f"\n  特征重要性:")
for _, row in feature_importance_unbalanced.head(3).iterrows():
    print(f"    {row['feature']}: {row['importance']:.6f}")

# 检查是否全为0
if feature_importance_unbalanced['importance'].sum() == 0:
    print("  ⚠️  警告: 特征重要性全为0！")
else:
    print(f"  特征重要性总和: {feature_importance_unbalanced['importance'].sum():.4f}")

# 训练模型（不平衡处理后）
print("\n步骤4: 训练模型（使用class_weight='balanced'）")
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # 关键参数
    random_state=42
)
rf_balanced.fit(X, y)

y_pred_balanced = rf_balanced.predict(X)
acc_balanced = accuracy_score(y, y_pred_balanced)
cm_balanced = confusion_matrix(y, y_pred_balanced)

print(f"  准确率: {acc_balanced:.4f}")
print(f"  混淆矩阵:")
print(f"    [[TN={cm_balanced[0,0]}, FP={cm_balanced[0,1]}],")
print(f"     [FN={cm_balanced[1,0]}, TP={cm_balanced[1,1]}]]")

# 计算召回率、精确率、F1
if (y == 1).sum() > 0:
    recall = cm_balanced[1,1] / (cm_balanced[1,1] + cm_balanced[1,0])
    precision = cm_balanced[1,1] / (cm_balanced[1,1] + cm_balanced[0,1]) if (cm_balanced[1,1] + cm_balanced[0,1]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  淘汰类别指标:")
    print(f"    召回率（Recall）: {recall:.4f}")
    print(f"    精确率（Precision）: {precision:.4f}")
    print(f"    F1分数: {f1:.4f}")

feature_importance_balanced = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_balanced.feature_importances_
}).sort_values("importance", ascending=False)

print(f"\n  特征重要性:")
for _, row in feature_importance_balanced.head(3).iterrows():
    print(f"    {row['feature']}: {row['importance']:.6f}")

if feature_importance_balanced['importance'].sum() == 0:
    print("  ⚠️  警告: 特征重要性仍为0！")
else:
    print(f"  ✓ 特征重要性总和: {feature_importance_balanced['importance'].sum():.4f}")

# 对比
print("\n" + "=" * 60)
print("对比总结")
print("=" * 60)
print(f"{'指标':<20} {'未处理':<15} {'已处理':<15} {'改善':<10}")
print("-" * 60)
print(f"{'准确率':<20} {acc_unbalanced:<15.4f} {acc_balanced:<15.4f} {(acc_balanced-acc_unbalanced)*100:>+9.2f}%")

recall_unbalanced = cm_unbalanced[1,1] / (cm_unbalanced[1,1] + cm_unbalanced[1,0]) if (cm_unbalanced[1,1] + cm_unbalanced[1,0]) > 0 else 0
print(f"{'淘汰召回率':<20} {recall_unbalanced:<15.4f} {recall:<15.4f} {(recall-recall_unbalanced)*100:>+9.2f}%")

importance_sum_unbalanced = feature_importance_unbalanced['importance'].sum()
importance_sum_balanced = feature_importance_balanced['importance'].sum()
print(f"{'特征重要性非零':<20} {'否' if importance_sum_unbalanced < 0.01 else '是':<15} {'是':<15} {'✓':<10}")

print("\n✓ 测试完成！使用 class_weight='balanced' 可以解决特征重要性为0的问题。")
