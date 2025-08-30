import pandas as pd
import os
from datetime import datetime

print("📊 ANALYZING FOOTBALL-DATA.CO.UK E0 FILES")
print("="*50)

# List all E0 files
data_dir = '/Users/maxime/Desktop/Oddsy/data/raw/'
e0_files = [f for f in os.listdir(data_dir) if f.startswith('E0') and f.endswith('.csv')]
e0_files.sort()

print(f"Found {len(e0_files)} E0 files:")
for f in e0_files:
    print(f"  • {f}")

# Analyze each file
file_info = {}

for filename in e0_files:
    filepath = os.path.join(data_dir, filename)
    try:
        df = pd.read_csv(filepath)
        
        # Get first date to identify season
        first_date = pd.to_datetime(df['Date'].iloc[0], format='%d/%m/%Y')
        season_year = first_date.year if first_date.month >= 8 else first_date.year - 1
        season = f"{season_year}-{str(season_year+1)[-2:]}"
        
        file_info[filename] = {
            'season': season,
            'matches': len(df),
            'columns': len(df.columns),
            'first_date': first_date.strftime('%Y-%m-%d'),
            'sample_teams': f"{df['HomeTeam'].iloc[0]} vs {df['AwayTeam'].iloc[0]}"
        }
        
        print(f"\n📁 {filename}:")
        print(f"   Season: {season}")
        print(f"   Matches: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        print(f"   Sample: {df['HomeTeam'].iloc[0]} vs {df['AwayTeam'].iloc[0]}")
        
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")

print(f"\n🔍 COLUMN ANALYSIS (first file):")
if e0_files:
    sample_df = pd.read_csv(os.path.join(data_dir, e0_files[0]))
    print(f"All columns ({len(sample_df.columns)}):")
    
    # Group columns by category
    core_cols = ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    stats_cols = [col for col in sample_df.columns if col in ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
    odds_cols = [col for col in sample_df.columns if any(bookie in col for bookie in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC'])]
    
    print(f"\n📋 CORE COLUMNS ({len(core_cols)}):")
    for col in core_cols:
        if col in sample_df.columns:
            print(f"   ✅ {col}")
    
    print(f"\n⚽ MATCH STATS COLUMNS ({len(stats_cols)}):")
    stats_mapping = {
        'HS': 'Home Shots', 'AS': 'Away Shots',
        'HST': 'Home Shots on Target', 'AST': 'Away Shots on Target',
        'HF': 'Home Fouls', 'AF': 'Away Fouls',
        'HC': 'Home Corners', 'AC': 'Away Corners',
        'HY': 'Home Yellow Cards', 'AY': 'Away Yellow Cards',
        'HR': 'Home Red Cards', 'AR': 'Away Red Cards'
    }
    
    for col in stats_cols:
        meaning = stats_mapping.get(col, col)
        print(f"   ✅ {col} = {meaning}")
    
    print(f"\n💰 BETTING ODDS COLUMNS: {len(odds_cols)} (showing first 10)")
    for col in odds_cols[:10]:
        print(f"   • {col}")
    
    # Check for possession
    possession_cols = [col for col in sample_df.columns if 'poss' in col.lower() or 'possession' in col.lower()]
    print(f"\n🎯 POSSESSION COLUMNS: {len(possession_cols)}")
    if possession_cols:
        for col in possession_cols:
            print(f"   ✅ {col}")
    else:
        print("   ❌ NO POSSESSION DATA FOUND")

print(f"\n🔄 COMPATIBILITY CHECK WITH ODDSY DATASET:")
# Load our existing dataset for comparison
oddsy_df = pd.read_csv('/Users/maxime/Desktop/Oddsy/data/raw/premier_league_2019_2024.csv')

print("Comparing team names...")
if e0_files:
    fd_df = pd.read_csv(os.path.join(data_dir, e0_files[0]))
    fd_teams = set(pd.concat([fd_df['HomeTeam'], fd_df['AwayTeam']]).unique())
    oddsy_teams = set(pd.concat([oddsy_df['HomeTeam'], oddsy_df['AwayTeam']]).unique())
    
    common_teams = fd_teams & oddsy_teams
    fd_only = fd_teams - oddsy_teams
    oddsy_only = oddsy_teams - fd_teams
    
    print(f"   Common teams: {len(common_teams)}")
    print(f"   Football-data only: {len(fd_only)} - {list(fd_only)[:3]}...")
    print(f"   Oddsy only: {len(oddsy_only)} - {list(oddsy_only)[:3]}...")

print(f"\n📝 RECOMMENDED FILE RENAMING:")
for filename, info in file_info.items():
    new_name = f"football_data_{info['season'].replace('-', '_')}.csv"
    print(f"   {filename} → {new_name}")

print(f"\n💡 NEXT STEPS:")
print("1. ❌ No possession data in football-data.co.uk files")
print("2. ✅ Rich match stats available (shots, corners, cards, fouls)")
print("3. ✅ Multiple betting odds sources") 
print("4. 🔄 Need to merge/reconcile team name differences")
print("5. 📁 Rename files with proper season names")

print(f"\n🎯 FOR ODDSY:")
print("• Football-data files complement our dataset well")
print("• More betting odds sources for validation")
print("• Same core stats (shots, corners) but different format")
print("• Still need to find possession data elsewhere")
print("• Could use these for cross-validation of existing features")