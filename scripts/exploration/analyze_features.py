import pandas as pd
import numpy as np
from math import sqrt

# Load processed dataset
df = pd.read_csv('/Users/maxime/Desktop/Oddsy/data/processed/premier_league_2019_2024_processed.csv')

print("🌳 ODDSY RANDOM FOREST FEATURE ANALYSIS")
print("="*50)

# Identify all potential features (exclude target and identifiers)
exclude_cols = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult', 
                'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals']

all_features = [col for col in df.columns if col not in exclude_cols]

print(f"📊 FEATURE POOL ANALYSIS:")
print(f"Total columns: {len(df.columns)}")
print(f"Excluded (target/id): {len(exclude_cols)}")
print(f"Available features: {len(all_features)}")

print(f"\n🎯 FEATURES CATEGORIES:")

# Core engineered features (0-1 normalized)
core_features = ['home_form', 'away_form', 'form_diff_normalized', 
                'elo_diff_normalized', 'h2h_score', 'home_advantage']

# Raw Elo ratings
elo_raw = ['home_elo_before', 'away_elo_before']

# Match stats
match_stats = ['HomeTeamShots', 'AwayTeamShots', 'HomeTeamCorners', 'AwayTeamCorners']

print(f"Core engineered (0-1): {len(core_features)} features")
for f in core_features:
    if f in df.columns:
        print(f"  • {f}")

print(f"\nRaw Elo: {len(elo_raw)} features")
for f in elo_raw:
    if f in df.columns:
        print(f"  • {f}")

print(f"\nMatch stats: {len(match_stats)} features")  
for f in match_stats:
    if f in df.columns:
        print(f"  • {f}")

# Check for additional features
other_features = [f for f in all_features if f not in core_features + elo_raw + match_stats]
if other_features:
    print(f"\nOther features: {len(other_features)}")
    for f in other_features:
        print(f"  • {f}")

print(f"\n🌲 RANDOM FOREST RECOMMENDATIONS:")

total_features = len([f for f in all_features if f in df.columns])
print(f"Total available features: {total_features}")

# Random Forest hyperparameter suggestions
print(f"\n📐 HYPERPARAMETER SUGGESTIONS:")

# max_features (features per split)
sqrt_features = int(sqrt(total_features))
print(f"max_features options:")
print(f"  • 'sqrt': {sqrt_features} features per split (default, recommended)")
print(f"  • 'log2': {int(np.log2(total_features))} features per split") 
print(f"  • None: {total_features} features per split (all features)")
print(f"  • Custom: 3-5 features (good for small feature set)")

# Feature importance analysis
print(f"\n🎯 EXPECTED FEATURE IMPORTANCE:")
print(f"Most important (likely):")
print(f"  1. elo_diff_normalized (team strength difference)")
print(f"  2. form_diff_normalized (recent form difference)") 
print(f"  3. home_advantage (always important in football)")
print(f"  4. h2h_score (historical matchups)")
print(f"  5. home_form / away_form (individual team form)")

print(f"\n🔧 FEATURE SELECTION STRATEGY:")
print(f"Option 1 - Minimal (6 core features):")
minimal_set = [f for f in core_features if f in df.columns]
print(f"  {minimal_set}")

print(f"\nOption 2 - Extended (core + raw elo):")
extended_set = [f for f in core_features + elo_raw if f in df.columns] 
print(f"  Features: {len(extended_set)}")

print(f"\nOption 3 - Full (all features):")
full_set = [f for f in all_features if f in df.columns]
print(f"  Features: {len(full_set)}")

print(f"\n💡 RECOMMENDATION FOR ODDSY:")
print(f"Start with Option 1 (6 core features) because:")
print(f"  • Well-engineered and normalized")
print(f"  • No redundancy")  
print(f"  • Easy to interpret")
print(f"  • max_features='sqrt' → {int(sqrt(6))} = 2-3 features per split")
print(f"  • Perfect for MVP validation")

# Sample first few rows of key features
print(f"\n📋 SAMPLE FEATURE VALUES:")
sample_features = ['elo_diff_normalized', 'form_diff_normalized', 'h2h_score', 'home_advantage']
available_sample = [f for f in sample_features if f in df.columns]
if available_sample:
    print(df[['HomeTeam', 'AwayTeam', 'FullTimeResult'] + available_sample].head(3).round(3).to_string())