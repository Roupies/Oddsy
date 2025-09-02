#!/usr/bin/env python3
"""
Dataset Summary & 564 Matches Mystery Investigation
Comprehensive audit of dataset composition, temporal splits, and sealed test integrity
"""
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def analyze_raw_data_sources():
    """Analyze all raw data sources to understand dataset composition"""
    logger = setup_logging()
    logger.info("=== 📊 RAW DATA SOURCES ANALYSIS ===")
    
    # Find all raw data files
    raw_patterns = [
        'data/raw/football_data_backup/*.csv',
        'data/raw/*.csv',
        'data/processed/premier_league_*.csv'
    ]
    
    all_files = []
    for pattern in raw_patterns:
        all_files.extend(glob.glob(pattern))
    
    logger.info(f"Found {len(all_files)} data files")
    
    raw_summary = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            # Parse dates if available
            date_col = None
            for col in ['Date', 'date', 'DATE']:
                if col in df.columns:
                    date_col = col
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    break
            
            # Extract seasons if available
            seasons = []
            if 'Season' in df.columns:
                seasons = df['Season'].dropna().unique().tolist()
            elif date_col:
                # Derive seasons from dates
                df['Year'] = df[date_col].dt.year
                seasons = sorted(df['Year'].dropna().unique().tolist())
            
            file_info = {
                'file': os.path.basename(file_path),
                'full_path': file_path,
                'rows': len(df),
                'columns': len(df.columns),
                'date_column': date_col,
                'min_date': df[date_col].min() if date_col else None,
                'max_date': df[date_col].max() if date_col else None,
                'seasons': seasons[:10],  # Limit to avoid clutter
                'has_teams': 'HomeTeam' in df.columns or 'Home' in df.columns,
                'sample_columns': df.columns.tolist()[:15]  # First 15 columns
            }
            
            raw_summary.append(file_info)
            
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
    
    # Sort by file name for consistent reporting
    raw_summary.sort(key=lambda x: x['file'])
    
    logger.info(f"\n📁 RAW DATA FILES SUMMARY:")
    logger.info("=" * 80)
    
    total_matches = 0
    for info in raw_summary:
        logger.info(f"\n📄 {info['file']}:")
        logger.info(f"  Rows: {info['rows']:,}")
        logger.info(f"  Columns: {info['columns']}")
        logger.info(f"  Date range: {info['min_date']} → {info['max_date']}")
        logger.info(f"  Seasons: {info['seasons']}")
        logger.info(f"  Has teams: {info['has_teams']}")
        
        if info['has_teams']:
            total_matches += info['rows']
    
    logger.info(f"\n📊 SUMMARY:")
    logger.info(f"Total potential matches across all files: {total_matches:,}")
    
    return raw_summary

def analyze_processed_datasets():
    """Analyze processed datasets used in modeling"""
    logger = setup_logging()
    logger.info("\n=== 🔧 PROCESSED DATASETS ANALYSIS ===")
    
    # Key datasets to analyze
    datasets_to_check = [
        'data/processed/v13_complete_with_dates.csv',
        'data/processed/premier_league_ml_ready.csv',
        'data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv',
        'data/processed/v13_production_dataset_encoded.csv'
    ]
    
    processed_summary = []
    
    for dataset_path in datasets_to_check:
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset not found: {dataset_path}")
            continue
            
        try:
            df = pd.read_csv(dataset_path)
            
            # Parse dates if available
            date_col = None
            for col in ['Date', 'date', 'DATE']:
                if col in df.columns:
                    date_col = col
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    break
            
            # Analyze temporal distribution
            temporal_info = {}
            if date_col:
                temporal_info = {
                    'min_date': df[date_col].min(),
                    'max_date': df[date_col].max(),
                    'date_range_days': (df[date_col].max() - df[date_col].min()).days,
                    'unique_dates': df[date_col].nunique()
                }
                
                # Season analysis
                if 'Season' in df.columns:
                    season_counts = df['Season'].value_counts().sort_index()
                    temporal_info['season_distribution'] = season_counts.to_dict()
                else:
                    # Derive seasons from dates
                    df['Year'] = df[date_col].dt.year
                    year_counts = df['Year'].value_counts().sort_index()
                    temporal_info['year_distribution'] = year_counts.to_dict()
            
            dataset_info = {
                'file': os.path.basename(dataset_path),
                'full_path': dataset_path,
                'rows': len(df),
                'columns': len(df.columns),
                'date_column': date_col,
                'temporal_info': temporal_info,
                'feature_columns': [col for col in df.columns if col not in ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult']],
                'has_target': 'FullTimeResult' in df.columns or 'target' in df.columns
            }
            
            processed_summary.append(dataset_info)
            
        except Exception as e:
            logger.error(f"Failed to analyze {dataset_path}: {e}")
    
    # Report processed datasets
    logger.info(f"\n🔧 PROCESSED DATASETS:")
    logger.info("=" * 80)
    
    for info in processed_summary:
        logger.info(f"\n📊 {info['file']}:")
        logger.info(f"  Rows: {info['rows']:,}")
        logger.info(f"  Columns: {info['columns']}")
        logger.info(f"  Date column: {info['date_column']}")
        logger.info(f"  Has target: {info['has_target']}")
        
        if info['temporal_info']:
            temp = info['temporal_info']
            logger.info(f"  Date range: {temp.get('min_date')} → {temp.get('max_date')}")
            logger.info(f"  Span: {temp.get('date_range_days', 0)} days")
            
            # Distribution
            dist = temp.get('season_distribution', temp.get('year_distribution', {}))
            if dist:
                logger.info(f"  Distribution: {dist}")
        
        if info['feature_columns']:
            logger.info(f"  Features ({len(info['feature_columns'])}): {info['feature_columns'][:8]}{'...' if len(info['feature_columns']) > 8 else ''}")
    
    return processed_summary

def investigate_564_matches_mystery():
    """Investigate the specific 564 matches in sealed test"""
    logger = setup_logging()
    logger.info("\n=== 🔍 564 MATCHES MYSTERY INVESTIGATION ===")
    
    # Load the dataset used in sealed test
    dataset_path = 'data/processed/v13_complete_with_dates.csv'
    if not os.path.exists(dataset_path):
        logger.error(f"Main dataset not found: {dataset_path}")
        return None
    
    df = pd.read_csv(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Loaded main dataset: {len(df)} total matches")
    logger.info(f"Date range: {df['Date'].min()} → {df['Date'].max()}")
    
    # Replicate the exact split used in sealed test
    cutoff_date = '2024-01-01'
    train_dev_data = df[df['Date'] < cutoff_date].copy()
    sealed_test_data = df[df['Date'] >= cutoff_date].copy()
    
    logger.info(f"\n📊 SEALED TEST SPLIT ANALYSIS:")
    logger.info(f"Cutoff date: {cutoff_date}")
    logger.info(f"Training data: {len(train_dev_data)} matches")
    logger.info(f"Sealed test: {len(sealed_test_data)} matches ← THE 564 MATCHES")
    
    # Detailed breakdown of sealed test period
    logger.info(f"\n🔍 SEALED TEST DETAILED BREAKDOWN:")
    logger.info(f"Date range: {sealed_test_data['Date'].min()} → {sealed_test_data['Date'].max()}")
    logger.info(f"Time span: {(sealed_test_data['Date'].max() - sealed_test_data['Date'].min()).days} days")
    
    # Monthly breakdown
    sealed_test_data['YearMonth'] = sealed_test_data['Date'].dt.to_period('M')
    monthly_counts = sealed_test_data['YearMonth'].value_counts().sort_index()
    
    logger.info(f"\n📅 MONTHLY BREAKDOWN (sealed test):")
    for period, count in monthly_counts.items():
        logger.info(f"  {period}: {count} matches")
    
    # Season breakdown if available
    if 'Season' in sealed_test_data.columns:
        season_counts = sealed_test_data['Season'].value_counts().sort_index()
        logger.info(f"\n🏆 SEASON BREAKDOWN (sealed test):")
        for season, count in season_counts.items():
            logger.info(f"  {season}: {count} matches")
    else:
        # Derive seasons from dates
        sealed_test_data['Year'] = sealed_test_data['Date'].dt.year
        year_counts = sealed_test_data['Year'].value_counts().sort_index()
        logger.info(f"\n📊 YEAR BREAKDOWN (sealed test):")
        for year, count in year_counts.items():
            logger.info(f"  {year}: {count} matches")
    
    # Check if 564 = expected for time period
    logger.info(f"\n🧮 MATCH DENSITY ANALYSIS:")
    
    # Premier League typically has ~380 matches per season
    days_in_period = (sealed_test_data['Date'].max() - sealed_test_data['Date'].min()).days
    expected_seasons = days_in_period / 365.25
    expected_matches_simple = expected_seasons * 380
    
    logger.info(f"Period: {days_in_period} days ({expected_seasons:.2f} seasons)")
    logger.info(f"Expected matches (380/season): {expected_matches_simple:.0f}")
    logger.info(f"Actual matches: {len(sealed_test_data)}")
    logger.info(f"Difference: {len(sealed_test_data) - expected_matches_simple:.0f}")
    
    # Check for data quality issues
    logger.info(f"\n🔍 DATA QUALITY CHECK (sealed test):")
    logger.info(f"Missing dates: {sealed_test_data['Date'].isnull().sum()}")
    logger.info(f"Duplicate dates: {sealed_test_data['Date'].duplicated().sum()}")
    
    if 'HomeTeam' in sealed_test_data.columns and 'AwayTeam' in sealed_test_data.columns:
        unique_teams = set(sealed_test_data['HomeTeam'].unique()) | set(sealed_test_data['AwayTeam'].unique())
        logger.info(f"Unique teams: {len(unique_teams)}")
        logger.info(f"Teams: {sorted(unique_teams)}")
        
        # Check for non-Premier League teams
        premier_league_teams = {
            'Arsenal', 'Aston Villa', 'Brighton', 'Burnley', 'Chelsea', 'Crystal Palace',
            'Everton', 'Fulham', 'Liverpool', 'Luton', 'Man City', 'Man United',
            'Newcastle', 'Nott\'m Forest', 'Sheffield United', 'Tottenham', 'West Ham', 'Wolves',
            'Bournemouth', 'Brentford', 'Leicester', 'Leeds', 'Southampton', 'Watford'
        }
        
        non_pl_teams = unique_teams - premier_league_teams
        if non_pl_teams:
            logger.warning(f"Non-Premier League teams found: {non_pl_teams}")
    
    # Sample of matches
    logger.info(f"\n📋 SAMPLE MATCHES (first 5 and last 5):")
    sample_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult'] if all(col in sealed_test_data.columns for col in ['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult']) else ['Date'] + [col for col in sealed_test_data.columns if col != 'Date'][:3]
    
    logger.info("First 5 matches:")
    for _, row in sealed_test_data[sample_columns].head().iterrows():
        logger.info(f"  {row.to_dict()}")
    
    logger.info("Last 5 matches:")
    for _, row in sealed_test_data[sample_columns].tail().iterrows():
        logger.info(f"  {row.to_dict()}")
    
    return {
        'total_matches': len(sealed_test_data),
        'date_range': (sealed_test_data['Date'].min(), sealed_test_data['Date'].max()),
        'period_days': days_in_period,
        'expected_matches': expected_matches_simple,
        'monthly_breakdown': monthly_counts.to_dict(),
        'unique_teams': len(unique_teams) if 'HomeTeam' in sealed_test_data.columns else None
    }

def compare_with_expected_seasons():
    """Compare actual data with expected Premier League structure"""
    logger = setup_logging()
    logger.info("\n=== ⚽ PREMIER LEAGUE STRUCTURE COMPARISON ===")
    
    # Premier League facts
    pl_facts = {
        'teams_per_season': 20,
        'matches_per_team': 38,  # Each team plays every other team twice (19*2)
        'total_matches_per_season': 380,  # 20 teams * 38 matches / 2
        'season_start_month': 8,  # August
        'season_end_month': 5,   # May
        'typical_season_months': 10
    }
    
    logger.info(f"Premier League structure:")
    for key, value in pl_facts.items():
        logger.info(f"  {key}: {value}")
    
    # Analyze our dataset against this structure
    dataset_path = 'data/processed/v13_complete_with_dates.csv'
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check season structure
        if 'Season' in df.columns:
            logger.info(f"\n📊 DATASET SEASON ANALYSIS:")
            season_counts = df['Season'].value_counts().sort_index()
            
            for season, count in season_counts.items():
                expected = pl_facts['total_matches_per_season']
                difference = count - expected
                status = "✅ CORRECT" if abs(difference) < 10 else f"⚠️ DIFF: {difference:+d}"
                logger.info(f"  {season}: {count} matches ({status})")
        
        # Check team counts
        if 'HomeTeam' in df.columns:
            unique_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
            logger.info(f"\nTeam analysis:")
            logger.info(f"  Unique teams across all seasons: {len(unique_teams)}")
            
            # Teams per season
            if 'Season' in df.columns:
                for season in df['Season'].unique():
                    season_data = df[df['Season'] == season]
                    season_teams = set(season_data['HomeTeam'].unique()) | set(season_data['AwayTeam'].unique())
                    logger.info(f"  {season}: {len(season_teams)} teams")

def generate_dataset_summary_report():
    """Generate comprehensive dataset summary report"""
    logger = setup_logging()
    logger.info("\n=== 📋 GENERATING COMPREHENSIVE REPORT ===")
    
    # Run all analyses
    raw_summary = analyze_raw_data_sources()
    processed_summary = analyze_processed_datasets()
    mystery_results = investigate_564_matches_mystery()
    compare_with_expected_seasons()
    
    # Generate markdown report
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    report_path = f'reports/dataset_summary_{timestamp}.md'
    
    os.makedirs('reports', exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(f"# Dataset Summary Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Executive Summary\n\n")
        
        if mystery_results:
            f.write(f"**564 Matches Mystery Resolution:**\n")
            f.write(f"- Sealed test contains {mystery_results['total_matches']} matches\n")
            f.write(f"- Date range: {mystery_results['date_range'][0]} to {mystery_results['date_range'][1]}\n")
            f.write(f"- Period: {mystery_results['period_days']} days\n")
            f.write(f"- Expected matches: {mystery_results['expected_matches']:.0f}\n")
            f.write(f"- Actual vs Expected: {mystery_results['total_matches'] - mystery_results['expected_matches']:.0f} difference\n\n")
        
        f.write(f"## Raw Data Sources ({len(raw_summary)} files)\n\n")
        for info in raw_summary:
            f.write(f"### {info['file']}\n")
            f.write(f"- Rows: {info['rows']:,}\n")
            f.write(f"- Date range: {info['min_date']} → {info['max_date']}\n")
            f.write(f"- Seasons: {info['seasons']}\n\n")
        
        f.write(f"## Processed Datasets ({len(processed_summary)} files)\n\n")
        for info in processed_summary:
            f.write(f"### {info['file']}\n")
            f.write(f"- Rows: {info['rows']:,}\n")
            f.write(f"- Features: {len(info['feature_columns'])}\n")
            if info['temporal_info']:
                f.write(f"- Date range: {info['temporal_info'].get('min_date')} → {info['temporal_info'].get('max_date')}\n")
            f.write(f"\n")
    
    logger.info(f"📄 Report generated: {report_path}")
    
    return {
        'report_path': report_path,
        'raw_files': len(raw_summary),
        'processed_files': len(processed_summary),
        'mystery_solved': mystery_results is not None
    }

if __name__ == "__main__":
    try:
        logger = setup_logging()
        logger.info("🔍 Starting comprehensive dataset analysis...")
        
        results = generate_dataset_summary_report()
        
        print(f"\n📊 DATASET ANALYSIS COMPLETE")
        print(f"Raw data files analyzed: {results['raw_files']}")
        print(f"Processed datasets analyzed: {results['processed_files']}")
        print(f"564 matches mystery: {'✅ INVESTIGATED' if results['mystery_solved'] else '❌ UNSOLVED'}")
        print(f"Report: {results['report_path']}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()