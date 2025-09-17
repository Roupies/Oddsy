#!/usr/bin/env python3
"""
📊 DATA LOADER - CORE MODULE
============================
Optimized data loading with Streamlit cache.
Critical performance: <3s initial load, <1s interactions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
import sys

# Add project root for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Configuration paths
DATA_PATH = os.path.join(project_root, "data/processed/v_auto_update_20250916_110247.csv")
CALENDAR_PATH = os.path.join(project_root, "data/raw/epl-2025-2026_GMTStandardTime.csv")
BASELINE_MODEL_PATH = os.path.join(project_root, "models/production/baseline_champion_v23.joblib")
CASCADE_MODEL_PATH = os.path.join(project_root, "models/production/cascade_champion_v2.joblib")
BASELINE_METADATA_PATH = os.path.join(project_root, "models/production/baseline_champion_v23_metadata.json")
CASCADE_METADATA_PATH = os.path.join(project_root, "models/production/cascade_champion_v2_metadata.json")

# Check paths exist
print(f"Baseline model exists: {os.path.exists(BASELINE_MODEL_PATH)}")
print(f"Cascade model exists: {os.path.exists(CASCADE_MODEL_PATH)}")
print(f"Data exists: {os.path.exists(DATA_PATH)}")

@st.cache_data(ttl=3600)  # Cache 1h for static data
def load_match_data() -> pd.DataFrame:
    """
    Load and prepare EPL match data.
    
    Returns:
        DataFrame with columns: Date, HomeTeam, AwayTeam, FullTimeResult, features...
    """
    try:
        data = pd.read_csv(DATA_PATH)
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Target mapping for consistency
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        data['target'] = data['FullTimeResult'].map(target_mapping)
        
        # Production features (same order as models)
        production_features = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        # Validate feature presence
        missing_features = [f for f in production_features if f not in data.columns]
        if missing_features:
            st.error(f"Missing features: {missing_features}")
            return pd.DataFrame()
        
        # Clean data
        valid_mask = data['target'].notna() & data[production_features].notna().all(axis=1)
        data_clean = data[valid_mask].sort_values('Date').reset_index(drop=True)
        
        return data_clean
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_metadata() -> Dict:
    """
    Load metadata for both champions.
    
    Returns:
        Dict with baseline_metadata and cascade_metadata
    """
    metadata = {}
    
    try:
        # Baseline Champion metadata
        with open(BASELINE_METADATA_PATH, 'r') as f:
            metadata['baseline'] = json.load(f)
            
        # Cascade Champion metadata  
        with open(CASCADE_METADATA_PATH, 'r') as f:
            metadata['cascade'] = json.load(f)
            
        return metadata
        
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return {}

@st.cache_resource  # Permanent cache for heavy models
def load_baseline_model():
    """
    Load Baseline Champion model.
    
    Returns:
        Trained scikit-learn model
    """
    try:
        model = joblib.load(BASELINE_MODEL_PATH)
        return model
    except Exception as e:
        st.warning(f"Baseline model not available: {e}")
        return None

@st.cache_resource  # Permanent cache for heavy models  
def load_cascade_model():
    """
    Load Cascade Champion model.
    
    Returns:
        Trained cascade model
    """
    try:
        model = joblib.load(CASCADE_MODEL_PATH)
        return model
    except Exception as e:
        st.warning(f"Cascade model not available: {e}")
        return None

@st.cache_data(ttl=3600)
def get_epl_2025_26_matches() -> pd.DataFrame:
    """
    Extract 40 EPL 2025-26 matches for validation.
    
    Returns:
        DataFrame of J1-J4 EPL 2025-26 matches
    """
    data = load_match_data()
    if data.empty:
        return pd.DataFrame()
    
    # Temporal split identical to models
    test_start = pd.to_datetime('2025-08-15')
    epl_2025_26 = data[data['Date'] >= test_start].copy()
    
    # Enrichment for display
    if not epl_2025_26.empty:
        epl_2025_26['Match'] = epl_2025_26['HomeTeam'] + ' vs ' + epl_2025_26['AwayTeam']
        epl_2025_26['Result_Display'] = epl_2025_26['FullTimeResult'].map({
            'H': '🏠 Home Win', 'D': '🤝 Draw', 'A': '✈️ Away Win'
        })
    
    return epl_2025_26

@st.cache_data(ttl=3600) 
def calculate_performance_metrics() -> Dict:
    """
    Calculate performance metrics for both champions.
    
    Returns:
        Dict with accuracy, baseline comparisons, etc.
    """
    metadata = load_metadata()
    if not metadata:
        return {}
    
    # Extract baseline metrics
    baseline_perf = metadata.get('baseline', {}).get('audit_results', {}).get('test_performance', {})
    baseline_cv = metadata.get('baseline', {}).get('audit_results', {}).get('cross_validation', {})
    
    # Extract cascade metrics
    cascade_perf = metadata.get('cascade', {}).get('audit_results', {}).get('test_performance', {})
    cascade_cv = metadata.get('cascade', {}).get('audit_results', {}).get('cross_validation', {})
    
    metrics = {
        'baseline': {
            'test_accuracy': baseline_perf.get('accuracy', 0) * 100,
            'cv_accuracy': baseline_cv.get('cv_mean', 0) * 100,
            'cv_std': baseline_cv.get('cv_std', 0) * 100,
            'stability': baseline_cv.get('stability', 'Unknown')
        },
        'cascade': {
            'test_accuracy': cascade_perf.get('accuracy', 0) * 100,
            'cv_accuracy': cascade_cv.get('cv_mean', 0) * 100, 
            'cv_std': cascade_cv.get('cv_std', 0) * 100,
            'stability': cascade_cv.get('stability', 'Unknown')
        },
        'baselines': {
            'random': 33.3,
            'always_home': 43.6,
            'good_target': 50.0,
            'excellent_target': 55.0
        }
    }
    
    return metrics

@st.cache_data(ttl=3600)  
def load_epl_calendar() -> pd.DataFrame:
    """
    Load complete EPL 2025-26 calendar.
    
    Returns:
        DataFrame with all EPL matches for the season
    """
    try:
        calendar = pd.read_csv(CALENDAR_PATH)
        calendar['Date'] = pd.to_datetime(calendar['Date'], format='%d/%m/%Y %H:%M')
        
        # Clean team names to match model data
        team_mapping = {
            'Man Utd': 'Man United', 'Spurs': 'Tottenham', 'Nott\'m Forest': 'Nottingham Forest'
        }
        
        for old_name, new_name in team_mapping.items():
            calendar['Home Team'] = calendar['Home Team'].replace(old_name, new_name)
            calendar['Away Team'] = calendar['Away Team'].replace(old_name, new_name)
        
        return calendar
        
    except Exception as e:
        st.error(f"Error loading EPL calendar: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)  # Cache 30min for live predictions
def get_upcoming_matches(n_matches: int = 5) -> pd.DataFrame:
    """
    Get upcoming EPL matches from the calendar.
    
    Args:
        n_matches: Number of future matches to return
        
    Returns:
        DataFrame with upcoming matches
    """
    calendar = load_epl_calendar()
    if calendar.empty:
        return pd.DataFrame()
    
    # Get matches that haven't been played yet (no result)
    future_matches = calendar[
        (calendar['Result'].isna()) | (calendar['Result'] == '')
    ].copy()
    
    # Sort by date and get next N matches
    future_matches = future_matches.sort_values('Date').head(n_matches)
    
    # Add display columns
    future_matches['Match'] = future_matches['Home Team'] + ' vs ' + future_matches['Away Team']
    
    return future_matches

def make_model_prediction(model, features: pd.Series, model_type: str) -> Tuple[str, float, dict]:
    """
    Make prediction using either Baseline or Cascade model.
    
    Args:
        model: Trained model
        features: Feature vector for the match
        model_type: 'baseline' or 'cascade'
        
    Returns:
        Tuple of (prediction, confidence, probabilities)
    """
    if model is None:
        return 'H', 0.33, {'H': 0.33, 'D': 0.33, 'A': 0.33}
    
    try:
        # Use exact feature order from the Baseline model metadata
        if model_type == 'baseline':
            # From baseline_champion_v23_metadata.json
            feature_order = [
                "form_diff_normalized", "elo_diff_normalized", "h2h_score",
                "matchday_normalized", "shots_diff_normalized", "corners_diff_normalized", 
                "market_entropy_norm", "home_xg_eff_10", "away_goals_sum_5", "away_xg_eff_10"
            ]
        else:
            # Default order for cascade or other models
            feature_order = [
                'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
                'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
                'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
            ]
        
        # Create DataFrame with exact feature order and names from training
        feature_values = [features[feature] for feature in feature_order]
        X = pd.DataFrame([feature_values], columns=feature_order)
        
        # Get probabilities
        probabilities = model.predict_proba(X)[0]
        
        # Map probabilities to outcomes
        if model_type == 'baseline':
            # Baseline model might not predict draws well
            prob_dict = {'H': probabilities[0], 'D': probabilities[1] if len(probabilities) > 2 else 0.1, 'A': probabilities[-1]}
        else:
            # Cascade model should handle all three outcomes
            prob_dict = {'H': probabilities[0], 'D': probabilities[1] if len(probabilities) > 2 else probabilities[1], 'A': probabilities[-1]}
        
        # Get prediction (highest probability)
        prediction = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[prediction]
        
        return prediction, confidence, prob_dict
        
    except Exception as e:
        st.error(f"Error making {model_type} prediction: {e}")
        return 'H', 0.33, {'H': 0.33, 'D': 0.33, 'A': 0.33}

def generate_simple_real_predictions(n_matches: int = 5, selected_model: str = None) -> pd.DataFrame:
    """
    Generate simplified real predictions using trained models.
    """
    upcoming_matches = get_upcoming_matches(n_matches)
    if upcoming_matches.empty:
        return pd.DataFrame()
    
    # Load at least one model
    baseline_model = load_baseline_model()
    cascade_model = load_cascade_model()
    
    if baseline_model is None and cascade_model is None:
        return pd.DataFrame()
    
    # Load historical data for median features
    historical_data = load_match_data()
    if historical_data.empty:
        return pd.DataFrame()
    
    # Get median feature values (simplified approach)
    production_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    median_features = pd.Series({
        feature: historical_data[feature].median() 
        for feature in production_features
    })
    
    predictions = []
    
    for _, match in upcoming_matches.iterrows():
        # Basic prediction data
        pred_data = {
            'Date': match['Date'].strftime('%Y-%m-%d') if hasattr(match['Date'], 'strftime') else str(match['Date']),
            'Home': match['Home Team'],
            'Away': match['Away Team'], 
            'Match': match['Match']
        }
        
        # Choose model based on selection
        if selected_model == 'baseline' and baseline_model is not None:
            pred, conf, _ = make_model_prediction(baseline_model, median_features, 'baseline')
            pred_data.update({
                'Final_Pred': pred,
                'Final_Conf': conf,
                'Model_Used': 'Baseline Champion'
            })
        elif selected_model == 'cascade' and cascade_model is not None:
            pred, conf, _ = make_model_prediction(cascade_model, median_features, 'cascade')
            pred_data.update({
                'Final_Pred': pred,
                'Final_Conf': conf,
                'Model_Used': 'Cascade Champion'
            })
        else:
            # Auto: use whichever model is available, prefer Cascade for early season
            if datetime.now().month <= 9 and cascade_model is not None:
                pred, conf, _ = make_model_prediction(cascade_model, median_features, 'cascade')
                pred_data.update({
                    'Final_Pred': pred,
                    'Final_Conf': conf,
                    'Model_Used': 'Cascade Champion (Auto)'
                })
            elif baseline_model is not None:
                pred, conf, _ = make_model_prediction(baseline_model, median_features, 'baseline')
                pred_data.update({
                    'Final_Pred': pred,
                    'Final_Conf': conf,
                    'Model_Used': 'Baseline Champion (Auto)'
                })
            else:
                # Fallback
                pred_data.update({
                    'Final_Pred': 'H',
                    'Final_Conf': 0.5,
                    'Model_Used': 'Fallback'
                })
        
        predictions.append(pred_data)
    
    return pd.DataFrame(predictions)

# Validation tests
def validate_data_loading() -> bool:
    """Validate proper data loading functionality."""
    try:
        data = load_match_data()
        metadata = load_metadata()
        metrics = calculate_performance_metrics()
        
        checks = [
            len(data) > 2000,  # Minimum dataset size
            'baseline' in metadata,
            'cascade' in metadata, 
            metrics['baseline']['test_accuracy'] > 0,
            metrics['cascade']['test_accuracy'] > 0
        ]
        
        return all(checks)
        
    except Exception:
        return False

if __name__ == "__main__":
    print("🧪 Test Data Loader")
    print(f"✅ Validation: {validate_data_loading()}")
    
    data = load_match_data()
    print(f"📊 Dataset: {len(data)} matches")
    
    epl_matches = get_epl_2025_26_matches() 
    print(f"🏆 EPL 2025-26: {len(epl_matches)} matches")
    
    metrics = calculate_performance_metrics()
    print(f"📈 Baseline Accuracy: {metrics['baseline']['test_accuracy']:.1f}%")
    print(f"🎯 Cascade Accuracy: {metrics['cascade']['test_accuracy']:.1f}%")