import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('/Users/maxime/Desktop/Oddsy/data/raw/PremierLeague.csv')

print("=== DATASET OVERVIEW ===")
print(f"Dataset shape: {df.shape}")
print(f"Total matches: {len(df)}")

print("\n=== COLUMNS ===")
print("Column names:")
for i, col in enumerate(df.columns.tolist(), 1):
    print(f"{i:2d}. {col}")

print("\n=== SAMPLE DATA ===")
print("First 3 rows:")
print(df.head(3).to_string())

print("\n=== SEASON DISTRIBUTION ===")
season_counts = df['Season'].value_counts().sort_index()
print(season_counts)

print("\n=== DATE RANGE ===")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

print("\n=== KEY COLUMNS INFO ===")
print("FullTimeResult distribution:")
print(df['FullTimeResult'].value_counts())

print("\n=== MISSING VALUES CHECK ===")
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
print("Columns with missing values:")
print(missing_data)