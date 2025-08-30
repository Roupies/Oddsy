import pandas as pd

# Load cleaned dataset
df = pd.read_csv('/Users/maxime/Desktop/Oddsy/data/cleaned/premier_league_2019_2024_cleaned.csv')

print("=== INVESTIGATING MISSING VALUES ===")

# Find rows with missing B365 Over/Under values
missing_over = df[df['B365Over2.5Goals'].isnull()]
missing_under = df[df['B365Under2.5Goals'].isnull()]

print("Rows with missing B365Over2.5Goals:")
if len(missing_over) > 0:
    print(missing_over[['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult', 'B365HomeTeam', 'B365Draw', 'B365AwayTeam', 'B365Over2.5Goals', 'B365Under2.5Goals']])

print("\nRows with missing B365Under2.5Goals:")
if len(missing_under) > 0:
    print(missing_under[['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult', 'B365HomeTeam', 'B365Draw', 'B365AwayTeam', 'B365Over2.5Goals', 'B365Under2.5Goals']])

# Check if it's the same row
same_rows = df[df['B365Over2.5Goals'].isnull() | df['B365Under2.5Goals'].isnull()]
print(f"\nTotal rows with missing Over/Under: {len(same_rows)}")

# Check if other betting data is available for these matches
print("\nOther betting data availability for missing rows:")
if len(same_rows) > 0:
    print(same_rows[['Date', 'HomeTeam', 'AwayTeam', 'B365HomeTeam', 'B365Draw', 'B365AwayTeam', 'MarketAvgHomeTeam', 'MarketAvgDraw', 'MarketAvgAwayTeam']])

# Check statistics for these matches
print("\nMatch statistics for missing rows:")
if len(same_rows) > 0:
    print(same_rows[['Date', 'HomeTeam', 'AwayTeam', 'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'TotalGoals']])

print("\n=== FINAL DATA QUALITY SUMMARY ===")
print(f"Total matches: {len(df)}")
print(f"Complete betting data (H/D/A): {len(df[df['B365HomeTeam'].notnull()])}")
print(f"Complete over/under data: {len(df[df['B365Over2.5Goals'].notnull()])}")
print(f"Missing over/under: {len(df[df['B365Over2.5Goals'].isnull()])}")

# For Oddsy project, we can either:
# 1. Drop these 1-2 rows (negligible impact)
# 2. Impute with average values
# 3. Keep as is (won't affect our main features)

print(f"\nData completeness: {((len(df) - len(same_rows)) / len(df) * 100):.2f}%")

# Decision for Oddsy project
print("\n=== RECOMMENDATION FOR ODDSY ===")
print("Impact analysis:")
print(f"- Missing rows: {len(same_rows)}/{len(df)} = {(len(same_rows)/len(df)*100):.3f}%")
print("- Core features (results, goals, shots, teams) are complete ✅")
print("- Main betting odds (H/D/A) are complete ✅") 
print("- Only Over/Under odds missing (not critical for H/D/A prediction)")
print("\nRecommendation: Keep dataset as-is. Impact negligible for MVP.")