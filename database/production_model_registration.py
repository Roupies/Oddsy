#!/usr/bin/env python3
"""
🏆 Production Model Registration
==============================

Register the 46% Cascade Champion as the production model
with complete metadata, feature requirements, and deployment info.
"""

import sys
import os
from datetime import datetime
import json
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.python_connector import OddsyDatabase

def register_cascade_champion_46():
    """Register the 46% Cascade Champion as production model"""
    print("🏆 REGISTERING CASCADE CHAMPION v2.1 - 46% ACCURACY")
    print("=" * 60)
    
    try:
        # Connect to database
        db = OddsyDatabase()
        print("✅ Connected to Oddsy PostgreSQL database")
        
        # Champion model metadata
        model_name = "Cascade Champion"
        model_version = "v2.1_production_46"
        
        # Performance metrics from our testing
        metrics = {
            'accuracy': 0.46,
            'precision_home': 0.48,  # From final test results
            'precision_draw': 0.25,
            'precision_away': 0.29,
            'recall_home': 0.52,
            'recall_draw': 0.14,  
            'recall_away': 0.38,
            'f1_score': 0.34,  # Macro average
            'total_predictions': 50,
            'correct_predictions': 23
        }
        
        # Hyperparameters that achieved 46%
        hyperparameters = {
            "architecture": "2_stage_cascade",
            "draw_weight": 2.5,
            "draw_threshold": 0.4,
            "n_estimators_stage1": 200,
            "n_estimators_stage2": 150,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "class_weight_stage1": {"non_draw": 1, "draw": 2.5},
            "class_weight_stage2": "balanced",
            "random_state": 42,
            "optimization_date": "2025-01-14",
            "test_dataset": "EPL_2025_26_50_matches"
        }
        
        # Feature importance from the 46% model
        feature_importance = {
            "elo_diff_normalized": 0.306,
            "market_entropy_norm": 0.251, 
            "shots_diff_normalized": 0.194,
            "home_xg_eff_10": 0.194,
            "away_xg_eff_10": 0.180,
            "form_diff_normalized": 0.156,
            "corners_diff_normalized": 0.145,
            "h2h_score": 0.124,
            "matchday_normalized": 0.089,
            "away_goals_sum_5": 0.067
        }
        
        print(f"📊 Registering model performance:")
        print(f"   Model: {model_name} {model_version}")
        print(f"   Accuracy: {metrics['accuracy']:.1%}")
        print(f"   Test dataset: EPL 2025-26 (50 matches)")
        
        # Save model performance to database
        db.save_model_performance(
            model_name=model_name,
            model_version=model_version, 
            metrics=metrics,
            dataset_used="EPL_2025_26_50_matches_mandatory",
            hyperparameters=hyperparameters
        )
        
        print("✅ Model performance saved to database")
        
        # Add feature importance as separate JSON (extend hyperparameters)
        extended_hyperparams = hyperparameters.copy()
        extended_hyperparams['feature_importance'] = feature_importance
        extended_hyperparams['production_features'] = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized', 
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score', 
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        # Update with extended metadata
        update_query = """
        UPDATE model_performance 
        SET hyperparameters = %s
        WHERE model_name = %s AND model_version = %s
        """
        
        db.execute_non_query(
            update_query, 
            (json.dumps(extended_hyperparams), model_name, model_version)
        )
        
        print("✅ Extended metadata (features & importance) saved")
        
        # Verify registration
        performance_data = db.execute_query(
            "SELECT * FROM model_performance WHERE model_name = %s AND model_version = %s",
            (model_name, model_version)
        )
        
        if len(performance_data) > 0:
            print(f"\n🎯 REGISTRATION VERIFICATION:")
            row = performance_data.iloc[0]
            print(f"   Model: {row['model_name']} {row['model_version']}")
            print(f"   Accuracy: {row['accuracy']:.1%}")
            print(f"   Total Predictions: {row['total_predictions']}")
            print(f"   Correct Predictions: {row['correct_predictions']}")
            print(f"   Evaluation Date: {row['evaluation_date']}")
            print(f"   Hyperparameters stored: ✅")
        
        print(f"\n✅ CASCADE CHAMPION v2.1 REGISTERED FOR PRODUCTION")
        print(f"🎯 Ready for deployment with 46% accuracy on EPL 2025-26")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Error registering model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_production_feature_requirements():
    """Create table for production feature requirements"""
    print(f"\n🔧 CREATING PRODUCTION FEATURE REQUIREMENTS")
    print("=" * 50)
    
    try:
        db = OddsyDatabase()
        
        # Create production_features table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS production_features (
            feature_id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            feature_name VARCHAR(100) NOT NULL,
            feature_type VARCHAR(20) NOT NULL CHECK (feature_type IN ('required', 'optional')),
            importance_score DECIMAL(6,4),
            default_value DECIMAL(8,4),
            preprocessing_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(model_name, model_version, feature_name)
        );
        """
        
        db.execute_non_query(create_table_sql)
        print("✅ production_features table created")
        
        # Insert feature requirements for Cascade Champion v2.1
        features_data = [
            ('Cascade Champion', 'v2.1_production_46', 'elo_diff_normalized', 'required', 0.306, 0.5, 'ELO difference normalized to 0-1 range'),
            ('Cascade Champion', 'v2.1_production_46', 'market_entropy_norm', 'required', 0.251, 0.5, 'Market entropy normalized, best draw predictor'),
            ('Cascade Champion', 'v2.1_production_46', 'shots_diff_normalized', 'required', 0.194, 0.5, 'Shots difference normalized'),
            ('Cascade Champion', 'v2.1_production_46', 'home_xg_eff_10', 'required', 0.194, 0.5, 'Home xG efficiency over 10 games'),
            ('Cascade Champion', 'v2.1_production_46', 'away_xg_eff_10', 'required', 0.180, 0.5, 'Away xG efficiency over 10 games'),
            ('Cascade Champion', 'v2.1_production_46', 'form_diff_normalized', 'required', 0.156, 0.5, 'Form difference normalized'),
            ('Cascade Champion', 'v2.1_production_46', 'corners_diff_normalized', 'required', 0.145, 0.5, 'Corners difference normalized'),
            ('Cascade Champion', 'v2.1_production_46', 'h2h_score', 'required', 0.124, 0.5, 'Head-to-head historical score'),
            ('Cascade Champion', 'v2.1_production_46', 'matchday_normalized', 'required', 0.089, 0.5, 'Matchday normalized to 0-1'),
            ('Cascade Champion', 'v2.1_production_46', 'away_goals_sum_5', 'required', 0.067, 0.5, 'Away goals sum over 5 games')
        ]
        
        insert_sql = """
        INSERT INTO production_features 
        (model_name, model_version, feature_name, feature_type, importance_score, default_value, preprocessing_notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_name, model_version, feature_name) DO UPDATE SET
            importance_score = EXCLUDED.importance_score,
            default_value = EXCLUDED.default_value,
            preprocessing_notes = EXCLUDED.preprocessing_notes
        """
        
        with db.conn.cursor() as cursor:
            for feature_data in features_data:
                cursor.execute(insert_sql, feature_data)
            db.conn.commit()
        
        print(f"✅ {len(features_data)} production features registered")
        
        # Verify feature requirements
        features_check = db.execute_query("""
            SELECT feature_name, feature_type, importance_score, default_value
            FROM production_features 
            WHERE model_name = 'Cascade Champion' AND model_version = 'v2.1_production_46'
            ORDER BY importance_score DESC
        """)
        
        print(f"\n📋 PRODUCTION FEATURES REGISTERED:")
        for _, row in features_check.iterrows():
            print(f"   {row['feature_name']}: {row['importance_score']:.3f} ({row['feature_type']})")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating feature requirements: {str(e)}")
        return False

def extend_model_performance_schema():
    """Extend model_performance table with production deployment columns"""
    print(f"\n🔧 EXTENDING MODEL PERFORMANCE SCHEMA")
    print("=" * 50)
    
    try:
        db = OddsyDatabase()
        
        # Add production deployment columns
        alter_statements = [
            "ALTER TABLE model_performance ADD COLUMN IF NOT EXISTS deployment_date TIMESTAMP",
            "ALTER TABLE model_performance ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT FALSE",
            "ALTER TABLE model_performance ADD COLUMN IF NOT EXISTS model_file_path VARCHAR(500)",
            "ALTER TABLE model_performance ADD COLUMN IF NOT EXISTS feature_requirements JSONB",
            "ALTER TABLE model_performance ADD COLUMN IF NOT EXISTS production_notes TEXT"
        ]
        
        for statement in alter_statements:
            db.execute_non_query(statement)
        
        print("✅ Schema extended with production deployment columns")
        
        # Mark the 46% model as active for production
        update_production_sql = """
        UPDATE model_performance SET 
            deployment_date = CURRENT_TIMESTAMP,
            is_active = TRUE,
            model_file_path = 'models/production/cascade_champion_v21_final.joblib',
            feature_requirements = %s,
            production_notes = 'Champion model with 46%% accuracy on EPL 2025-26. Optimized through comprehensive feature testing.'
        WHERE model_name = 'Cascade Champion' AND model_version = 'v2.1_production_46'
        """
        
        feature_requirements = {
            "required_features": 10,
            "feature_list": [
                'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
                'corners_diff_normalized', 'form_diff_normalized', 'h2h_score', 
                'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
            ],
            "preprocessing": {
                "fill_missing_with": 0.5,
                "normalization": "0_to_1_range",
                "temporal_validation": "no_future_data_leakage"
            }
        }
        
        db.execute_non_query(update_production_sql, (json.dumps(feature_requirements),))
        print("✅ Cascade Champion v2.1 marked as active production model")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Error extending schema: {str(e)}")
        return False

def main():
    """Complete production model registration"""
    print("🚀 PRODUCTION MODEL REGISTRATION PIPELINE")
    print("=" * 60)
    
    success_count = 0
    
    # Step 1: Register the 46% model
    if register_cascade_champion_46():
        success_count += 1
    
    # Step 2: Create feature requirements table
    if create_production_feature_requirements():
        success_count += 1
    
    # Step 3: Extend schema for production
    if extend_model_performance_schema():
        success_count += 1
    
    print(f"\n" + "=" * 60)
    print(f"🎉 PRODUCTION REGISTRATION COMPLETE")
    print("=" * 60)
    print(f"✅ Steps completed: {success_count}/3")
    
    if success_count == 3:
        print(f"🏆 CASCADE CHAMPION v2.1 (46%) READY FOR PRODUCTION!")
        print(f"🎯 Model registered with complete metadata")
        print(f"📊 Feature requirements documented") 
        print(f"🔧 Production deployment flags set")
        print(f"💾 Database schema extended for production")
    else:
        print(f"⚠️ Some steps failed - check logs above")
    
    return success_count == 3

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 READY FOR PRODUCTION DEPLOYMENT!")
    else:
        print(f"\n❌ Registration incomplete - manual intervention needed")