"""
测试脚本：检查数据文件是否能正常加载
"""
import pandas as pd
import json
import sys

print("=" * 60)
print("Data File Check Script")
print("=" * 60)

# 1. 检查CSV文件（尝试多种编码）
print("\n1. Check dance_competition_final_processed.csv")
encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'iso-8859-1', 'utf-16']
csv_loaded = False
df = None

for encoding in encodings:
    try:
        df = pd.read_csv('dance_competition_final_processed.csv', encoding=encoding, encoding_errors='ignore')
        print(f"[OK] Loaded with {encoding} encoding")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {len(df.columns)}")
        print(f"  - First 5 columns: {df.columns[:5].tolist()}")
        csv_loaded = True
        break
    except Exception as e:
        print(f"[FAIL] {encoding} encoding failed")

if not csv_loaded:
    print("[ERROR] Cannot load CSV file, please check file format")
    sys.exit(1)

# 2. 检查JSON文件
print("\n2. Check fan_vote_constraints.json")
try:
    with open('fan_vote_constraints.json', 'r', encoding='utf-8') as f:
        constraints = json.load(f)
    print(f"[OK] Loaded JSON file successfully")
    print(f"  - Records: {len(constraints)}")
    if len(constraints) > 0:
        print(f"  - Keys in first record: {list(constraints[0].keys())}")
        print(f"  - Example season: {constraints[0].get('season', 'N/A')}")
        print(f"  - Example week: {constraints[0].get('week', 'N/A')}")
except Exception as e:
    print(f"[ERROR] JSON loading failed: {e}")
    sys.exit(1)

# 3. 检查特征文件
print("\n3. Check dance_competition_features.csv")
features_loaded = False
try:
    for encoding in encodings:
        try:
            features_df = pd.read_csv('dance_competition_features.csv', encoding=encoding, encoding_errors='ignore')
            print(f"[OK] Loaded with {encoding} encoding")
            print(f"  - Rows: {len(features_df)}")
            print(f"  - Columns: {len(features_df.columns)}")
            features_loaded = True
            break
        except:
            continue
    if not features_loaded:
        print("[WARN] Features file not loaded")
except Exception as e:
    print(f"[ERROR] Features file loading failed: {e}")

# 4. 检查必需列是否存在
print("\n4. Check required columns")
required_cols = [
    "season", "week", "celebrity_name", "in_competition",
    "non_competition_week", "week_eliminated", "eliminated_this_week",
    "relative_level"
]

if df is not None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[WARN] Missing columns: {missing_cols}")
        print(f"  Actual columns: {df.columns.tolist()}")
    else:
        print("[OK] All required columns exist")
    
    # 5. 数据样本预览
    print("\n5. Data sample preview")
    print(df.head(3))

print("\n" + "=" * 60)
print("Data check completed!")
print("=" * 60)
