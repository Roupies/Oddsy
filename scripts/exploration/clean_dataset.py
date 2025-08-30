import pandas as pd
import numpy as np
from datetime import datetime

# Load filtered dataset
df = pd.read_csv('/Users/maxime/Desktop/Oddsy/data/raw/premier_league_2019_2024.csv')

print("=== ODDSY DATA CLEANING ===")
print(f"Starting with: {len(df)} matches")

print("\n=== 1. DATA TYPES ANALYSIS ===")
print("Current data types:")
print(df.dtypes)

print("\n=== 2. TEAM NAMES STANDARDIZATION ===")
print("Unique team names:")
all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
print(sorted(all_teams))

# Check for potential team name inconsistencies
print("\n=== 3. CHECKING FOR INCONSISTENCIES ===")

# Check Date format
print("Date format check:")
print(f"Sample dates: {df['Date'].head(3).tolist()}")

# Check for duplicates
duplicates = df.duplicated(subset=['Date', 'HomeTeam', 'AwayTeam']).sum()
print(f"Duplicate matches: {duplicates}")

# Check result consistency
print("Result consistency:")
print(f"H results where home goals < away goals: {len(df[(df['FullTimeResult'] == 'H') & (df['FullTimeHomeTeamGoals'] < df['FullTimeAwayTeamGoals'])])}")
print(f"A results where away goals < home goals: {len(df[(df['FullTimeResult'] == 'A') & (df['FullTimeAwayTeamGoals'] < df['FullTimeHomeTeamGoals'])])}")
print(f"D results where goals are not equal: {len(df[(df['FullTimeResult'] == 'D') & (df['FullTimeHomeTeamGoals'] != df['FullTimeAwayTeamGoals'])])}")

# Check for negative values in stats
print("\n=== 4. NEGATIVE VALUES CHECK ===")
numeric_cols = ['FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'HomeTeamShots', 'AwayTeamShots']
for col in numeric_cols:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        print(f"{col}: {negative_count} negative values")

print("\n=== 5. DATA CLEANING PROCESS ===")

# Create a copy for cleaning
df_clean = df.copy()

# Convert Date to datetime
print("Converting Date to datetime...")
df_clean['Date'] = pd.to_datetime(df_clean['Date'])

# Sort by date to ensure chronological order
print("Sorting by date...")
df_clean = df_clean.sort_values('Date').reset_index(drop=True)

# Standardize team names (if needed)
# Check if there are variations like "Man City" vs "Manchester City"
team_mapping = {
    # Add mappings if inconsistencies found
}

if team_mapping:
    print("Applying team name standardization...")
    df_clean['HomeTeam'] = df_clean['HomeTeam'].replace(team_mapping)
    df_clean['AwayTeam'] = df_clean['AwayTeam'].replace(team_mapping)

# Create derived columns for validation
print("Adding validation columns...")
df_clean['GoalDifference'] = df_clean['FullTimeHomeTeamGoals'] - df_clean['FullTimeAwayTeamGoals']
df_clean['TotalGoals'] = df_clean['FullTimeHomeTeamGoals'] + df_clean['FullTimeAwayTeamGoals']

# Validate results consistency
def validate_result(row):
    if row['GoalDifference'] > 0:
        return 'H'
    elif row['GoalDifference'] < 0:
        return 'A'
    else:
        return 'D'

df_clean['ExpectedResult'] = df_clean.apply(validate_result, axis=1)
result_mismatches = (df_clean['FullTimeResult'] != df_clean['ExpectedResult']).sum()
print(f"Result validation: {result_mismatches} mismatches found")

if result_mismatches > 0:
    print("Mismatched results:")
    mismatches = df_clean[df_clean['FullTimeResult'] != df_clean['ExpectedResult']]
    print(mismatches[['Date', 'HomeTeam', 'AwayTeam', 'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'FullTimeResult', 'ExpectedResult']].head())

# Drop validation columns
df_clean = df_clean.drop(['ExpectedResult'], axis=1)

print("\n=== 6. FINAL CLEANING STATS ===")
print(f"Cleaned dataset shape: {df_clean.shape}")
print(f"Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
print(f"Seasons: {sorted(df_clean['Season'].unique())}")
print(f"Unique teams: {len(pd.concat([df_clean['HomeTeam'], df_clean['AwayTeam']]).unique())}")

print("\n=== 7. MISSING VALUES FINAL CHECK ===")
missing_final = df_clean.isnull().sum()
missing_final = missing_final[missing_final > 0]
if len(missing_final) > 0:
    print("Remaining missing values:")
    print(missing_final)
else:
    print("No missing values in key columns ✅")

# Save cleaned dataset
output_path = '/Users/maxime/Desktop/Oddsy/data/cleaned/premier_league_2019_2024_cleaned.csv'
df_clean.to_csv(output_path, index=False)
print(f"\n=== SAVED ===")
print(f"Cleaned dataset saved to: {output_path}")

print("\n=== SAMPLE CLEANED DATA ===")
print("First 3 rows:")
sample_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'FullTimeResult', 'GoalDifference', 'TotalGoals']
print(df_clean[sample_cols].head(3).to_string())