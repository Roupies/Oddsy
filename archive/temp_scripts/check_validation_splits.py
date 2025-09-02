import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime

print("🔍 ANALYSING TIMESERIESSPLIT VALIDATION STRATEGY")
print("=" * 60)

# Load the original dataset with dates
df_full = pd.read_csv('/Users/maxime/Desktop/Oddsy/data/processed/premier_league_ml_ready.csv')
df_full['Date'] = pd.to_datetime(df_full['Date'])

# Sort by date to understand chronological order
df_full = df_full.sort_values('Date').reset_index(drop=True)

print(f"Full dataset: {len(df_full)} matches")
print(f"Date range: {df_full['Date'].min()} → {df_full['Date'].max()}")

# Check available features in this dataset
print(f"Available columns: {list(df_full.columns)}")

# Use available features (adapt to this specific dataset)
available_features = [col for col in df_full.columns if col not in ['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult']]
print(f"Using features: {available_features}")

X = df_full[available_features].fillna(0.5)
label_mapping = {'H': 0, 'D': 1, 'A': 2}
y = df_full['FullTimeResult'].map(label_mapping)

# Analyze TimeSeriesSplit with 5 folds
tscv = TimeSeriesSplit(n_splits=5)

print(f"\n📊 TIMESERIESSPLIT ANALYSIS (5 folds):")
print("-" * 50)

fold = 1
all_test_indices = set()

for train_idx, test_idx in tscv.split(X):
    train_dates = df_full.iloc[train_idx]['Date']
    test_dates = df_full.iloc[test_idx]['Date']
    
    print(f"\nFold {fold}:")
    print(f"  Train: {len(train_idx)} samples ({train_dates.min()} → {train_dates.max()})")
    print(f"  Test:  {len(test_idx)} samples ({test_dates.min()} → {test_dates.max()})")
    
    # Check for temporal leakage
    if train_dates.max() > test_dates.min():
        print(f"  ⚠️ TEMPORAL LEAKAGE: Train max > Test min")
    else:
        print(f"  ✅ No temporal leakage")
    
    # Add test indices to track what's been "seen"
    all_test_indices.update(test_idx)
    
    fold += 1

# Check what percentage of data has been "tested" on
print(f"\n🔍 DATA LEAKAGE ANALYSIS:")
print("-" * 50)
tested_percentage = len(all_test_indices) / len(df_full) * 100
print(f"Total data tested during CV: {len(all_test_indices)}/{len(df_full)} ({tested_percentage:.1f}%)")

if tested_percentage > 80:
    print("❌ HIGH LEAKAGE: Most data has been 'seen' during validation")
    print("   → 53.05% performance is likely OPTIMISTIC")
elif tested_percentage > 60:
    print("⚠️ MODERATE LEAKAGE: Significant portion tested")
    print("   → Performance might be somewhat optimistic")
else:
    print("✅ LOW LEAKAGE: Most data unseen during training")

# Check if we have any truly unseen recent data
latest_test_date = max([df_full.iloc[list(test_idx)]['Date'].max() 
                       for _, test_idx in tscv.split(X)])

remaining_data = df_full[df_full['Date'] > latest_test_date]
print(f"\nUNSEEN DATA after all CV:")
if len(remaining_data) > 0:
    print(f"✅ {len(remaining_data)} matches never used in CV ({remaining_data['Date'].min()} → {remaining_data['Date'].max()})")
    print("   → True sealed test possible!")
else:
    print("❌ No unseen data remaining")
    print("   → All data has been used in cross-validation")

# Show season breakdown if available
if 'Season' in df_full.columns:
    print(f"\n📅 SEASON BREAKDOWN:")
    print("-" * 50)
    season_counts = df_full['Season'].value_counts().sort_index()
    for season, count in season_counts.items():
        season_data = df_full[df_full['Season'] == season]
        print(f"{season}: {count} matches ({season_data['Date'].min()} → {season_data['Date'].max()})")
else:
    print(f"\n📅 YEAR BREAKDOWN:")
    print("-" * 50)
    df_full['Year'] = df_full['Date'].dt.year
    year_counts = df_full['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        year_data = df_full[df_full['Year'] == year]
        print(f"{year}: {count} matches ({year_data['Date'].min()} → {year_data['Date'].max()})")

print(f"\n💡 CONCLUSION:")
print("-" * 50)
if tested_percentage > 80:
    print("🔴 RISK: TimeSeriesSplit has 'seen' most of the data")
    print("   → 53.05% result is cross-validation, NOT sealed test")
    print("   → True performance likely 1-3% lower")
    print("   → Need proper holdout on recent unseen data")
else:
    print("🟢 GOOD: TimeSeriesSplit respects temporal order")
    print("   → 53.05% is realistic estimate")
    print("   → But still not a true sealed test")