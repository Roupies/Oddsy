import pandas as pd
import numpy as np
import json
import os
from utils import setup_logging, load_config, validate_data_quality, version_data

def prepare_ml_data(config_path='config/config.json'):
    """
    Prepare ML-ready data with proper label encoding and X/y separation.
    
    Target encoding:
    - 0: Home win (H)
    - 1: Draw (D)
    - 2: Away win (A)
    """
    # Setup logging and load config
    logger = setup_logging(config_path)
    config = load_config(config_path)
    
    logger.info("Starting ML data preparation...")
    
    input_file = config['data']['input_file']
    output_dir = config['data']['output_dir']
    
    logger.info(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target distribution:")
    target_dist = df['FullTimeResult'].value_counts().sort_index()
    for target, count in target_dist.items():
        logger.info(f"  {target}: {count} ({count/len(df):.1%})")
    
    # Encode target labels
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    df['target'] = df['FullTimeResult'].map(label_mapping)
    
    # Verify encoding
    logger.info("Target encoding verification:")
    for original, encoded in label_mapping.items():
        count = (df['target'] == encoded).sum()
        logger.info(f"  {original} -> {encoded}: {count} samples")
    
    # Define feature columns (exclude non-ML columns)
    feature_columns = [
        'form_diff_normalized',
        'elo_diff_normalized', 
        'h2h_score',
        'home_advantage',
        'matchday_normalized',
        'season_period_numeric',
        'shots_diff_normalized',
        'corners_diff_normalized',
        'points_diff_normalized',
        'position_diff_normalized'
    ]
    
    # Separate features (X) and target (y)
    X = df[feature_columns].copy()
    y = df['target'].copy()
    
    # Validate data quality
    validation_passed = validate_data_quality(X, y, feature_columns, logger)
    if not validation_passed:
        raise ValueError("Data quality validation failed. Check logs for details.")
    
    logger.info(f"Data shapes - Features: {X.shape}, Target: {y.shape}")
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    
    # Save versioned data
    logger.info("Saving processed data with versioning...")
    
    # Save X and y separately with versioning
    X_file = version_data(X, f"{output_dir}X_features.csv", config)
    y_df = pd.DataFrame({'target': y})
    y_file = version_data(y_df, f"{output_dir}y_target.csv", config)
    
    # Save complete dataset with encoded target
    ml_dataset = df[feature_columns + ['target']].copy()
    ml_file = version_data(ml_dataset, f"{output_dir}premier_league_final_ml.csv", config)
    
    # Save configuration files
    os.makedirs('config/', exist_ok=True)
    
    # Save feature list
    with open('config/features.json', 'w') as f:
        json.dump(feature_columns, f, indent=2)
    
    # Save target mapping
    target_info = {
        'label_mapping': label_mapping,
        'reverse_mapping': {v: k for k, v in label_mapping.items()},
        'class_names': ['Home', 'Draw', 'Away'],
        'distribution': df['target'].value_counts().to_dict()
    }
    
    with open('config/target_mapping.json', 'w') as f:
        json.dump(target_info, f, indent=2)
    
    logger.info("Files saved:")
    logger.info(f"- Features: {X_file}")
    logger.info(f"- Target: {y_file}")
    logger.info(f"- Complete ML dataset: {ml_file}")
    logger.info(f"- Feature list: config/features.json")
    logger.info(f"- Target mapping: config/target_mapping.json")
    
    return X, y, feature_columns, target_info

if __name__ == "__main__":
    try:
        X, y, features, target_info = prepare_ml_data()
        
        logger = setup_logging()
        logger.info("=== ML Data Preparation Complete ===")
        logger.info(f"Features: {len(features)}")
        logger.info(f"Samples: {len(X)}")
        logger.info(f"Classes: {len(target_info['class_names'])}")
        logger.info("Ready for Random Forest training!")
        
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Data preparation failed: {str(e)}")
        raise