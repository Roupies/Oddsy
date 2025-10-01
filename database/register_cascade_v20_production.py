#!/usr/bin/env python3
"""
🏆 Register Cascade Champion v2.0 - Production (46%)
==================================================

Register the TRUE Cascade Champion v2.0 with 46% accuracy
as the production model, replacing any v2.1 versions.
"""

import sys
import os
from datetime import datetime
import json
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.python_connector import OddsyDatabase

def register_cascade_v20_production():
    """Register Cascade Champion v2.0 as THE production model"""
    print("🏆 REGISTERING CASCADE CHAMPION v2.0 - THE TRUE CHAMPION")
    print("=" * 60)
    
    try:
        db = OddsyDatabase()
        print("✅ Connected to Oddsy PostgreSQL database")
        
        # Deactivate any existing models first
        deactivate_query = """
        UPDATE model_performance 
        SET is_active = FALSE 
        WHERE model_name = 'Cascade Champion'
        """
        db.execute_non_query(deactivate_query)
        print("🔄 Deactivated previous Cascade Champion versions")
        
        # TRUE Cascade Champion v2.0 metadata
        model_name = "Cascade Champion"
        model_version = "v2.0_production"
        
        # Performance metrics (46% on EPL 2025-26)
        metrics = {
            'accuracy': 0.46,
            'precision_home': 0.56,  # From original v2.0 metadata
            'precision_draw': 0.33,
            'precision_away': 0.45,
            'recall_home': 0.60,
            'recall_draw': 0.33,  
            'recall_away': 0.45,
            'f1_score': 0.46,  # Weighted average
            'total_predictions': 50,
            'correct_predictions': 23
        }
        
        # TRUE v2.0 hyperparameters from metadata
        hyperparameters = {
            "architecture": "Cascade_Binary_Ternary",
            "stage_1": {
                "purpose": "Draw_Detection",
                "algorithm": "RandomForest",
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 5,
                "class_weight": {"non_draw": 1, "draw": 2.5}
            },
            "stage_2": {
                "purpose": "Home_Away_Classification", 
                "algorithm": "RandomForest",
                "n_estimators": 150,
                "class_weight": "balanced"
            },
            "cascade_logic": {
                "draw_threshold": 0.4,
                "draw_weight": 2.5
            },
            "random_state": 42,
            "version": "v2.0_cascade_dual_stage",
            "test_dataset": "EPL_2025_26_50_matches"
        }
        
        # TRUE v2.0 features (from metadata)
        production_features = [
            "elo_diff_normalized",
            "market_entropy_norm", 
            "shots_diff_normalized",
            "corners_diff_normalized",
            "form_diff_normalized",
            "h2h_score",
            "matchday_normalized",
            "home_xg_eff_10",
            "away_xg_eff_10",
            "away_goals_sum_5"
        ]
        
        # TRUE v2.0 feature importance (from metadata)
        feature_importance = {
            "elo_diff_normalized": 0.151,
            "market_entropy_norm": 0.148,
            "home_xg_eff_10": 0.115,
            "away_xg_eff_10": 0.108,
            "shots_diff_normalized": 0.104,
            "corners_diff_normalized": 0.093,
            "matchday_normalized": 0.081,
            "form_diff_normalized": 0.079,
            "h2h_score": 0.061,
            "away_goals_sum_5": 0.060
        }
        
        print(f"📊 Registering TRUE Cascade Champion v2.0:")
        print(f"   Model: {model_name} {model_version}")
        print(f"   Accuracy: {metrics['accuracy']:.1%} (EPL 2025-26)")
        print(f"   Architecture: 2-stage cascade")
        print(f"   Features: {len(production_features)}")
        
        # Save to database
        db.save_model_performance(
            model_name=model_name,
            model_version=model_version,
            metrics=metrics,
            dataset_used="EPL_2025_26_50_matches",
            hyperparameters=hyperparameters
        )
        print("✅ Cascade Champion v2.0 registered in database")
        
        # Add extended metadata
        extended_hyperparams = hyperparameters.copy()
        extended_hyperparams['feature_importance'] = feature_importance
        extended_hyperparams['production_features'] = production_features
        extended_hyperparams['is_true_champion'] = True
        
        # Update with complete metadata
        update_query = """
        UPDATE model_performance 
        SET hyperparameters = %s,
            deployment_date = CURRENT_TIMESTAMP,
            is_active = TRUE,
            model_file_path = 'models/production/cascade_champion_v20_production.joblib',
            feature_requirements = %s,
            production_notes = 'TRUE Cascade Champion v2.0 - Original champion with 46%% accuracy on EPL 2025-26'
        WHERE model_name = %s AND model_version = %s
        """
        
        feature_requirements = {
            "required_features": 10,
            "feature_list": production_features,
            "feature_importance": feature_importance,
            "preprocessing": {
                "fill_missing_with": 0.5,
                "cascade_stage1_features": production_features[:5],  # Enhanced features
                "cascade_stage2_features": production_features[5:]   # Classical features
            }
        }
        
        db.execute_non_query(
            update_query,
            (json.dumps(extended_hyperparams), json.dumps(feature_requirements), model_name, model_version)
        )
        print("✅ Extended metadata and production flags set")
        
        # Update production_features table
        print("🔧 Updating production features table...")
        
        # Clear previous features for this model
        clear_query = "DELETE FROM production_features WHERE model_name = %s AND model_version = %s"
        db.execute_non_query(clear_query, (model_name, model_version))
        
        # Insert TRUE v2.0 features
        features_data = []
        for i, feature_name in enumerate(production_features):
            importance = feature_importance.get(feature_name, 0.1)
            features_data.append((
                model_name, model_version, feature_name, 'required', 
                importance, 0.5, f'Cascade Champion v2.0 feature #{i+1}'
            ))
        
        insert_sql = """
        INSERT INTO production_features 
        (model_name, model_version, feature_name, feature_type, importance_score, default_value, preprocessing_notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        with db.conn.cursor() as cursor:
            for feature_data in features_data:
                cursor.execute(insert_sql, feature_data)
            db.conn.commit()
        
        print(f"✅ {len(features_data)} v2.0 features registered")
        
        # Verify registration
        verification_query = """
        SELECT model_name, model_version, accuracy, is_active, deployment_date
        FROM model_performance 
        WHERE model_name = %s AND model_version = %s
        """
        
        result = db.execute_query(verification_query, (model_name, model_version))
        
        if len(result) > 0:
            row = result.iloc[0]
            print(f"\n🎯 REGISTRATION VERIFIED:")
            print(f"   Model: {row['model_name']} {row['model_version']}")  
            print(f"   Accuracy: {row['accuracy']:.1%}")
            print(f"   Active: {row['is_active']}")
            print(f"   Deployed: {row['deployment_date']}")
        
        print(f"\n🏆 CASCADE CHAMPION v2.0 IS NOW THE ACTIVE PRODUCTION MODEL!")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Error registering Cascade v2.0: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 REGISTERING TRUE CASCADE CHAMPION v2.0")
    print("=" * 60)
    
    success = register_cascade_v20_production()
    
    if success:
        print(f"\n🎉 CASCADE CHAMPION v2.0 READY FOR PRODUCTION!")
        print(f"🏆 The TRUE champion (46%) is now active!")
    else:
        print(f"\n❌ Registration failed")