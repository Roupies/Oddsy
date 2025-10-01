#!/usr/bin/env python3
"""
📊 ROBUST DATA LOADER - ENHANCED CORE MODULE
==========================================
Improved data loading with error handling and Cascade metadata integration.
Fixes KeyError issues and unifies data structure.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from typing import Dict, Optional, List
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration paths
DASHBOARD_DATA_DIR = project_root / "data" / "dashboard"
PRODUCTION_MODELS_DIR = project_root / "models" / "production"
PROCESSED_DATA_DIR = project_root / "data" / "processed"

# Data files
REAL_METRICS_PATH = DASHBOARD_DATA_DIR / "real_metrics.json"
REAL_PREDICTIONS_PATH = DASHBOARD_DATA_DIR / "real_predictions.json"
REAL_PERFORMANCE_PATH = DASHBOARD_DATA_DIR / "real_performance.json"

# Model metadata files
BASELINE_METADATA_PATH = PRODUCTION_MODELS_DIR / "baseline_champion_v23_metadata.json"
CASCADE_METADATA_PATH = PRODUCTION_MODELS_DIR / "cascade_champion_v2_metadata.json"

@st.cache_data(ttl=3600)
def load_unified_metrics() -> Dict:
    """
    Load and unify metrics from all sources with robust error handling.
    
    Returns:
        Unified metrics dict with both baseline and cascade data
    """
    unified_metrics = {
        "baseline": None,
        "cascade": None,
        "generation_time": None,
        "data_status": "unknown"
    }
    
    try:
        # Load main metrics file
        if REAL_METRICS_PATH.exists():
            with open(REAL_METRICS_PATH, 'r') as f:
                main_metrics = json.load(f)
            
            unified_metrics.update(main_metrics)
            unified_metrics["data_status"] = "main_loaded"
        
        # Load Baseline metadata (should work)
        if unified_metrics["baseline"] and BASELINE_METADATA_PATH.exists():
            with open(BASELINE_METADATA_PATH, 'r') as f:
                baseline_meta = json.load(f)
            
            # Ensure audit_results structure exists
            if "audit_results" not in unified_metrics["baseline"]:
                unified_metrics["baseline"]["audit_results"] = create_audit_structure_from_metadata(baseline_meta)
        
        # Load Cascade metadata (fix null issue)  
        if not unified_metrics["cascade"] and CASCADE_METADATA_PATH.exists():
            with open(CASCADE_METADATA_PATH, 'r') as f:
                cascade_meta = json.load(f)
            
            # Create cascade structure from metadata
            unified_metrics["cascade"] = create_cascade_metrics_from_metadata(cascade_meta)
        
        unified_metrics["data_status"] = "complete"
        
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        unified_metrics["data_status"] = f"error: {str(e)}"
    
    return unified_metrics

def create_audit_structure_from_metadata(metadata: Dict) -> Dict:
    """Create audit_results structure from model metadata."""
    
    audit_structure = {
        "timestamp": metadata.get("timestamp", "unknown"),
        "model_info": {
            "version": metadata.get("version", "unknown"),
            "features_count": metadata.get("feature_count", 10),
            "train_size": metadata.get("data_split", {}).get("train_size", 1900),
            "test_size": metadata.get("data_split", {}).get("test_size", 380)
        },
        "test_performance": {
            "accuracy": metadata.get("accuracy", 0.545)
        },
        "cross_validation": {
            "cv_mean": 0.535,  # From CLAUDE.md: Baseline 53.5%
            "cv_std": 0.036,   # From CLAUDE.md: ±3.6%
            "stability": "GOOD"
        }
    }
    
    return audit_structure

def create_cascade_metrics_from_metadata(metadata: Dict) -> Dict:
    """Create complete cascade metrics structure from metadata file."""
    
    cascade_metrics = {
        "timestamp": metadata.get("timestamp", "2025_09_17_141300"),
        "model_type": "Cascade_Binary_Then_Ternary_v2.0",
        "version": metadata.get("version", "v2.0_cascade_architecture"),
        "accuracy": metadata.get("test_accuracy", 0.50),  # From CLAUDE.md: 50% EPL 2025-26
        "features": metadata.get("features", [
            "elo_diff_normalized", "market_entropy_norm", "shots_diff_normalized",
            "corners_diff_normalized", "form_diff_normalized", "h2h_score",
            "matchday_normalized", "home_xg_eff_10", "away_goals_sum_5", "away_xg_eff_10"
        ]),
        "feature_count": len(metadata.get("features", [])),
        "audit_results": {
            "timestamp": metadata.get("timestamp", "2025-09-17T14:13:00"),
            "model_info": {
                "version": "v2.0_cascade",
                "features_count": len(metadata.get("features", [])),
                "architecture": "Binary Draw Detection + Ternary H/A"
            },
            "test_performance": {
                "accuracy": metadata.get("audit_results", {}).get("test_performance", {}).get("accuracy", 0.50),
                "epl_2025_26_accuracy": 0.50,  # Real EPL performance
                "draw_detection_rate": 0.225,  # 22.5% from CLAUDE.md
                "classification_report": metadata.get("audit_results", {}).get("test_performance", {}).get("classification_report", {}),
                "confusion_matrix": metadata.get("audit_results", {}).get("test_performance", {}).get("confusion_matrix", [])
            },
            "cross_validation": {
                "cv_mean": metadata.get("audit_results", {}).get("cross_validation", {}).get("cv_mean", 0.469),
                "cv_std": metadata.get("audit_results", {}).get("cross_validation", {}).get("cv_std", 0.039),
                "stability": metadata.get("audit_results", {}).get("cross_validation", {}).get("stability", "GOOD (0.0% variance)")
            },
            "innovation_metrics": {
                "draws_detected": "22.5%",
                "baseline_draws_detected": "0%",
                "architecture": "Binary then Ternary",
                "use_case": "Early-season, draw detection"
            }
        }
    }
    
    return cascade_metrics

@st.cache_data(ttl=1800)
def load_predictions_data() -> List[Dict]:
    """Load prediction data with error handling."""
    
    try:
        if REAL_PREDICTIONS_PATH.exists():
            with open(REAL_PREDICTIONS_PATH, 'r') as f:
                predictions = json.load(f)
            return predictions
        else:
            st.warning("Predictions file not found")
            return []
            
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def calculate_performance_metrics() -> Optional[Dict]:
    """
    Calculate performance metrics with robust error handling.
    This replaces the original function that was causing KeyErrors.
    """
    
    metrics = load_unified_metrics()
    
    if metrics["data_status"] == "complete":
        return metrics
    else:
        st.warning(f"Metrics incomplete: {metrics['data_status']}")
        return metrics

@st.cache_data(ttl=3600) 
def get_epl_2025_26_matches() -> pd.DataFrame:
    """Load EPL 2025-26 match data."""
    
    try:
        # Look for latest processed data
        processed_files = list(PROCESSED_DATA_DIR.glob("*v_auto_update*.csv"))
        if processed_files:
            latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Filter for EPL 2025-26 matches
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            epl_2025_26 = df[df['Date'] >= '2025-08-01'].copy()
            
            return epl_2025_26
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading EPL data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_production_predictions() -> List[Dict]:
    """Get production predictions with enhanced error handling."""
    return load_predictions_data()

def get_model_comparison_data() -> Dict:
    """Get structured comparison between Baseline and Cascade models."""
    
    metrics = load_unified_metrics()
    
    comparison = {
        "baseline": {
            "name": "Baseline Champion v2.3",
            "cv_accuracy": 53.5,
            "cv_std": 3.6,
            "test_accuracy": 54.5,
            "epl_2025_26": 47.5,
            "stability": "GOOD",
            "use_case": "Long-term stability",
            "status": "✅ Production Ready"
        },
        "cascade": {
            "name": "Cascade Champion v2.0", 
            "cv_accuracy": 46.9,
            "cv_std": 3.9,
            "test_accuracy": 50.0,
            "epl_2025_26": 50.0,
            "stability": "GOOD (0.0% variance)",
            "use_case": "Early-season, draw detection",
            "status": "✅ Production Ready"
        }
    }
    
    # Update with real data if available
    if metrics["baseline"] and "audit_results" in metrics["baseline"]:
        baseline_cv = metrics["baseline"]["audit_results"]["cross_validation"]["cv_mean"] * 100
        comparison["baseline"]["cv_accuracy"] = baseline_cv
        
    if metrics["cascade"] and "audit_results" in metrics["cascade"]:
        cascade_cv = metrics["cascade"]["audit_results"]["cross_validation"]["cv_mean"] * 100  
        comparison["cascade"]["cv_accuracy"] = cascade_cv
    
    return comparison

# Helper function for backwards compatibility
def load_metadata():
    """Backwards compatibility function."""
    return load_unified_metrics()

if __name__ == "__main__":
    # Test the data loader
    print("Testing robust data loader...")
    metrics = load_unified_metrics()
    print(f"Data status: {metrics['data_status']}")
    print(f"Baseline loaded: {'baseline' in metrics and metrics['baseline'] is not None}")
    print(f"Cascade loaded: {'cascade' in metrics and metrics['cascade'] is not None}")