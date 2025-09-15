import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('/Users/maxime/Desktop/Oddsy/data/raw/PremierLeague.csv')

print("=== FILTERING 2019-2024 SEASONS ===")
target_seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']

# Filter for target seasons
df_filtered = df[df['Season'].isin(target_seasons)].copy()

print(f"Original dataset: {len(df)} matches")
print(f"Filtered dataset: {len(df_filtered)} matches")
print(f"Expected: ~2280 matches (380 × 6 seasons)")

print("\n=== SEASON BREAKDOWN ===")
season_breakdown = df_filtered['Season'].value_counts().sort_index()
print(season_breakdown)

print("\n=== SAMPLE FILTERED DATA ===")
print("First 3 rows of filtered data:")
print(df_filtered.head(3)[['MatchID', 'Season', 'Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult']].to_string())

print("\n=== DATA QUALITY CHECK ON FILTERED DATA ===")
print("FullTimeResult distribution:")
print(df_filtered['FullTimeResult'].value_counts())

print("\n=== MISSING VALUES IN KEY COLUMNS (FILTERED) ===")
key_columns = [
    'HomeTeam', 'AwayTeam', 'FullTimeResult', 
    'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals',
    'HomeTeamShots', 'AwayTeamShots', 
    'HomeTeamShotsOnTarget', 'AwayTeamShotsOnTarget',
    'HomeTeamCorners', 'AwayTeamCorners',
    'B365HomeTeam', 'B365Draw', 'B365AwayTeam'
]

print("Missing values in key columns:")
for col in key_columns:
    if col in df_filtered.columns:
        missing_count = df_filtered[col].isnull().sum()
        missing_pct = (missing_count / len(df_filtered)) * 100
        print(f"{col:25s}: {missing_count:4d} ({missing_pct:5.1f}%)")

print("\n=== UNIQUE TEAMS IN FILTERED DATA ===")
all_teams = pd.concat([df_filtered['HomeTeam'], df_filtered['AwayTeam']]).unique()
print(f"Number of unique teams: {len(all_teams)}")
print("Teams:", sorted(all_teams))

print("\n=== DATE RANGE FILTERED ===")
print(f"Date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")

# Save filtered dataset
output_path = '/Users/maxime/Desktop/Oddsy/data/raw/premier_league_2019_2024.csv'
df_filtered.to_csv(output_path, index=False)
print(f"\n=== SAVED ===")
print(f"Filtered dataset saved to: {output_path}")
print(f"Shape: {df_filtered.shape}")