#!/usr/bin/env python3
"""
Test v4.0 Algorithm Upgrade - XGBoost vs RandomForest
Evaluate if Gradient Boosting can improve on v3.1 efficiency breakthrough

Strategy: Test XGBoost, LightGBM, and CatBoost on same v3.1 feature set
Baseline: v3.1 RandomForest (56.28% accuracy)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available - install with: pip install catboost")

import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlgorithmUpgradeTester:
    """Test multiple algorithms on v3.1 dataset to find the best performer."""
    
    def __init__(self):
        # Best features from v3.1 efficiency breakthrough
        self.best_features = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'goalkeeping_advantage_10', 
            'away_goalkeeping_efficiency_10_normalized', 'goalkeeping_advantage_10_normalized',
            'net_performance_advantage_10_normalized', 'net_performance_advantage_10',
            'goalkeeping_advantage_5_normalized', 'away_xg_eff_10',
            'matchday_normalized', 'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
        ]
        
        # Model configurations optimized for football prediction
        self.models = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=10,
                    min_samples_leaf=5, random_state=42, n_jobs=-1
                ),
                'description': 'v3.1 Baseline - Proven performer'
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, eval_metric='mlogloss', verbosity=0
                ),
                'description': 'Gradient Boosting - Sequential error correction'
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=-1
                ),
                'description': 'Fast Gradient Boosting - Memory efficient'
            }
        }
        
        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = {
                'model': cb.CatBoostClassifier(
                    iterations=200, depth=6, learning_rate=0.1,
                    subsample=0.8, random_seed=42, verbose=False
                ),
                'description': 'Advanced Gradient Boosting - Categorical features'
            }
    
    def load_and_prepare_data(self):
        """Load v3.1 dataset and prepare for testing."""
        
        logger.info("Loading v3.1 efficiency dataset...")
        df = pd.read_csv('data/processed/v31_efficiency_features_2025_09_06.csv')
        
        logger.info(f"Loaded dataset: {df.shape}")
        
        # Filter available features
        available_features = [f for f in self.best_features if f in df.columns]
        logger.info(f"Available features: {len(available_features)}/{len(self.best_features)}")
        
        # Clean data
        df_clean = df.dropna(subset=available_features + ['FullTimeResult'])
        logger.info(f"Clean dataset: {df_clean.shape}")
        
        # Train/test split (same as v3.1 for fair comparison)
        split_idx = int(len(df_clean) * 0.8)
        df_train = df_clean[:split_idx]
        df_test = df_clean[split_idx:]
        
        # Prepare features and targets
        X_train = df_train[available_features]
        X_test = df_test[available_features]
        
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        y_train = df_train['FullTimeResult'].map(target_mapping)
        y_test = df_test['FullTimeResult'].map(target_mapping)
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, available_features
    
    def evaluate_algorithm(self, name, model_config, X_train, X_test, y_train, y_test):
        """Evaluate a single algorithm with comprehensive metrics."""
        
        logger.info(f"Testing {name}...")
        
        model = model_config['model']
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Core metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        logloss = log_loss(y_test, y_pred_proba)
        
        # Cross-validation for stability
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):  # CatBoost
            feature_importance = model.get_feature_importance()
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'log_loss': logloss,
            'cv_accuracy_mean': cv_mean,
            'cv_accuracy_std': cv_std,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def run_algorithm_comparison(self):
        """Run comprehensive algorithm comparison."""
        
        logger.info("🚀 Starting v4.0 Algorithm Upgrade Test...")
        
        # Load data
        X_train, X_test, y_train, y_test, available_features = self.load_and_prepare_data()
        
        # Test each algorithm
        results = {}
        
        for name, model_config in self.models.items():
            try:
                results[name] = self.evaluate_algorithm(
                    name, model_config, X_train, X_test, y_train, y_test
                )
                results[name]['description'] = model_config['description']
                results[name]['status'] = 'success'
            except Exception as e:
                logger.error(f"Failed to test {name}: {str(e)}")
                results[name] = {
                    'status': 'failed',
                    'error': str(e),
                    'description': model_config['description']
                }
        
        # Generate comprehensive report
        self.generate_comprehensive_report(results, y_test, available_features)
        
        return results
    
    def generate_comprehensive_report(self, results, y_test, features):
        """Generate detailed comparison report."""
        
        print("\n" + "="*80)
        print("🚀 v4.0 ALGORITHM UPGRADE EVALUATION")
        print("="*80)
        
        print(f"\n🎯 BASELINE COMPARISON:")
        print(f"   • v3.1 RandomForest Target: 56.28% (efficiency breakthrough)")
        print(f"   • Test Set Size: {len(y_test)} matches")
        print(f"   • Features Used: {len(features)} optimized features")
        
        # Performance ranking
        successful_results = {k: v for k, v in results.items() if v.get('status') == 'success'}
        ranked_results = sorted(successful_results.items(), 
                              key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"\n📊 ALGORITHM PERFORMANCE RANKING:")
        baseline_accuracy = None
        
        for i, (name, result) in enumerate(ranked_results, 1):
            accuracy = result['accuracy']
            f1_macro = result['f1_macro']
            logloss = result['log_loss']
            cv_mean = result['cv_accuracy_mean']
            cv_std = result['cv_accuracy_std']
            
            if name == 'RandomForest':
                baseline_accuracy = accuracy
                marker = "🎯 BASELINE"
                improvement = "0.00pp"
            else:
                improvement = f"{(accuracy - baseline_accuracy)*100:+.2f}pp" if baseline_accuracy else "N/A"
                if accuracy > baseline_accuracy:
                    marker = "🚀 UPGRADE"
                elif accuracy > baseline_accuracy - 0.005:  # Within 0.5pp
                    marker = "⚡ SIMILAR"
                else:
                    marker = "📉 LOWER"
            
            print(f"   {i}. {marker} {name}: {accuracy:.4f} ({accuracy*100:.2f}%) [{improvement}]")
            print(f"      • F1-Macro: {f1_macro:.3f}")
            print(f"      • Log Loss: {logloss:.3f}")
            print(f"      • CV: {cv_mean:.3f} ± {cv_std:.3f}")
            print(f"      • Description: {result['description']}")
        
        # Best performer analysis
        if ranked_results:
            best_name, best_result = ranked_results[0]
            best_accuracy = best_result['accuracy']
            
            print(f"\n🏆 BEST PERFORMER: {best_name.upper()}")
            print(f"   • Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            
            if baseline_accuracy and best_accuracy > baseline_accuracy:
                improvement = (best_accuracy - baseline_accuracy) * 100
                print(f"   • Improvement: +{improvement:.2f}pp over RandomForest")
                
                if improvement >= 1.0:
                    verdict = "🚀 MAJOR UPGRADE"
                    recommendation = "Deploy immediately - significant improvement"
                elif improvement >= 0.5:
                    verdict = "✅ SOLID UPGRADE"  
                    recommendation = "Deploy after validation - meaningful improvement"
                elif improvement >= 0.2:
                    verdict = "⚡ MARGINAL UPGRADE"
                    recommendation = "Consider deployment - small but positive improvement"
                else:
                    verdict = "📊 MINIMAL IMPROVEMENT"
                    recommendation = "Stay with RandomForest - improvement too small"
            else:
                improvement = 0
                verdict = "📊 NO IMPROVEMENT"
                recommendation = "Continue with RandomForest baseline"
            
            print(f"   • Verdict: {verdict}")
            print(f"   • Recommendation: {recommendation}")
        
        # Feature importance comparison (top algorithm vs baseline)
        if len(ranked_results) >= 2 and ranked_results[0][1].get('feature_importance') is not None:
            print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
            
            best_importance = ranked_results[0][1]['feature_importance']
            rf_result = next((r for name, r in results.items() if name == 'RandomForest'), None)
            
            if rf_result and rf_result.get('feature_importance') is not None:
                rf_importance = rf_result['feature_importance']
                
                print(f"   Top 10 Features - {ranked_results[0][0]} vs RandomForest:")
                
                # Get top features from best model
                feature_ranking = list(zip(features, best_importance))
                feature_ranking.sort(key=lambda x: x[1], reverse=True)
                
                for i, (feature, importance) in enumerate(feature_ranking[:10]):
                    rf_imp = rf_importance[features.index(feature)] if feature in features else 0
                    print(f"   {i+1:2d}. {feature}: {importance:.3f} (RF: {rf_imp:.3f})")
        
        # Algorithm-specific insights
        print(f"\n💡 ALGORITHM INSIGHTS:")
        
        for name, result in results.items():
            if result.get('status') != 'success':
                print(f"   • {name}: ❌ Failed - {result.get('error', 'Unknown error')}")
            else:
                if name == 'XGBoost':
                    print(f"   • XGBoost: Sequential error correction, good for complex patterns")
                elif name == 'LightGBM':  
                    print(f"   • LightGBM: Fast training, memory efficient, good for large datasets")
                elif name == 'CatBoost':
                    print(f"   • CatBoost: Handles categorical features well, robust to overfitting")
                elif name == 'RandomForest':
                    print(f"   • RandomForest: Stable baseline, less prone to overfitting")
        
        # Business implications
        print(f"\n💰 BUSINESS IMPLICATIONS:")
        if ranked_results and ranked_results[0][1]['accuracy'] > baseline_accuracy:
            best_acc = ranked_results[0][1]['accuracy'] * 100
            improvement = (ranked_results[0][1]['accuracy'] - baseline_accuracy) * 100
            print(f"   • Performance: {best_acc:.2f}% accuracy (+{improvement:.2f}pp)")
            print(f"   • ROI Impact: Each 1pp ≈ 2-3% betting ROI improvement")
            print(f"   • Expected ROI Gain: +{improvement*2.5:.1f}% betting performance")
            
            if improvement >= 1.0:
                print(f"   • Status: 🚀 BREAKTHROUGH - Major algorithmic advancement")
            elif improvement >= 0.5:
                print(f"   • Status: ✅ SUCCESS - Meaningful improvement achieved")
            else:
                print(f"   • Status: ⚡ MARGINAL - Small but positive gain")
        else:
            print(f"   • No improvement over RandomForest baseline")
            print(f"   • Recommendation: Continue with proven v3.1 architecture")
        
        # Next steps
        print(f"\n📋 NEXT STEPS:")
        if ranked_results and ranked_results[0][1]['accuracy'] > baseline_accuracy:
            best_name = ranked_results[0][0]
            print(f"   1. Deploy {best_name} as new v4.0 production algorithm")
            print(f"   2. Hyperparameter optimization for {best_name}")
            print(f"   3. Comprehensive validation on additional data")
            print(f"   4. Consider ensemble methods combining top performers")
        else:
            print(f"   1. Continue with RandomForest as proven solution")
            print(f"   2. Focus on feature engineering over algorithm changes")
            print(f"   3. Explore advanced features (fatigue, referee data)")
            print(f"   4. Consider ensemble methods if needed")

def main():
    """Execute v4.0 algorithm upgrade test."""
    
    logger.info("🚀 Starting v4.0 Algorithm Upgrade Test...")
    
    tester = AlgorithmUpgradeTester()
    results = tester.run_algorithm_comparison()
    
    logger.info("✅ v4.0 Algorithm Upgrade Test Complete!")
    return results

if __name__ == "__main__":
    main()