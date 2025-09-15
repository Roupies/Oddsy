#!/usr/bin/env python3
"""
Binary Anti-Bias Model with Optimized Metrics

Creates specialized Home/Not-Home binary classifier to combat extreme home bias
revealed in rolling validation. Uses F1-Score and Log-Loss optimization with
Group K-Fold validation to prevent team-specific overfitting.

Based on error profile analysis and concept drift findings.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, log_loss, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BinaryAntiBiasModel:
    """
    Binary classifier optimized to combat home bias with rigorous validation
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = None
        
        # Core features (drift-adjusted importance)
        self.core_features = [
            'elo_diff_normalized',
            'form_diff_normalized', 
            'market_entropy_norm',
            'home_xg_eff_10',
            'away_xg_eff_10',
            'shots_diff_normalized',
            'corners_diff_normalized',
            'h2h_score',
            'away_goals_sum_5'
        ]
        
        # New anti-bias features (to be engineered)
        self.antibias_features = []
        
        self.model = None
        self.scaler = None
        self.feature_importance = None
        
        # Target metrics (based on Gemini's suggestions)
        self.target_metrics = {
            'f1_score': 0.72,
            'log_loss': 0.65,
            'precision': 0.70,
            'recall': 0.75
        }
        
    def load_and_prepare_data(self):
        """Load dataset and prepare for binary classification"""
        
        print("📊 Loading and preparing data for binary anti-bias model...")
        
        # Load dataset
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        
        print(f"✅ Loaded dataset: {len(self.dataset)} matches")
        
        # Filter for established teams only (avoid promoted teams bias)
        promoted_teams = ['Leeds', 'Sunderland', 'Burnley']
        
        # Get established teams data from historical seasons
        established_data = self.dataset[
            (~self.dataset['HomeTeam'].isin(promoted_teams)) &
            (~self.dataset['AwayTeam'].isin(promoted_teams)) &
            (self.dataset['Season'] != '2025-2026')  # Exclude EPL 2025-26 for now
        ].copy()
        
        print(f"📈 Established teams historical data: {len(established_data)} matches")
        
        # Create binary target: Home Win (1) vs Not Home Win (0)
        established_data['binary_target'] = (established_data['FullTimeResult'] == 'H').astype(int)
        
        # Distribution analysis
        home_win_rate = established_data['binary_target'].mean()
        print(f"🎯 Binary target distribution:")
        print(f"   Home wins: {home_win_rate:.1%}")
        print(f"   Not home wins (D+A): {1-home_win_rate:.1%}")
        
        self.dataset = established_data
        
        return established_data
        
    def engineer_antibias_features(self):
        """Engineer specialized features to combat home bias based on error analysis"""
        
        print("\n🛠️ Engineering anti-bias features...")
        
        df = self.dataset.copy()
        
        # Feature 1: Uncertainty amplifier (from error analysis - high entropy = higher error rate)
        df['uncertainty_amplified'] = df['market_entropy_norm'] * (
            1 + abs(df['elo_diff_normalized'] - 0.5) * 2  # Amplify when close matches
        )
        
        # Feature 2: Elo confidence zones (from vulnerability analysis)
        df['elo_confidence_zone'] = np.select([
            (df['elo_diff_normalized'] >= 0.45) & (df['elo_diff_normalized'] <= 0.55),
            df['elo_diff_normalized'] < 0.45,
            df['elo_diff_normalized'] > 0.55
        ], [0, -1, 1], default=0)  # 0=danger zone, -1=away_favored, 1=home_favored
        
        # Feature 3: Away strength composite (combat away prediction weakness)
        df['away_strength_composite'] = (
            df['away_xg_eff_10'] * 0.4 +
            (1 - df['elo_diff_normalized']) * 0.3 +  # Away team relative strength
            (df['away_goals_sum_5'] / 10) * 0.3  # Away scoring form
        )
        
        # Feature 4: Home vulnerability (detect weak home performances)
        df['home_vulnerability'] = (
            (1 - df['home_xg_eff_10']) * 0.5 +  # Poor home xG efficiency
            df['market_entropy_norm'] * 0.5  # Market uncertainty
        )
        
        # Feature 5: Form momentum differential (enhanced from basic form_diff)
        df['form_momentum'] = df['form_diff_normalized'] * (
            1 + abs(df['form_diff_normalized'] - 0.5)  # Amplify strong form differences
        )
        
        # Feature 6: Historical disadvantage (detect teams that struggle at home)
        # Simplified proxy using current features
        df['historical_home_disadvantage'] = np.select([
            df['h2h_score'] < 0.4,  # Historical underperformance  
            df['h2h_score'] > 0.6,  # Historical overperformance
        ], [1, -1], default=0)
        
        self.antibias_features = [
            'uncertainty_amplified',
            'elo_confidence_zone', 
            'away_strength_composite',
            'home_vulnerability',
            'form_momentum',
            'historical_home_disadvantage'
        ]
        
        print(f"✅ Engineered {len(self.antibias_features)} anti-bias features:")
        for feature in self.antibias_features:
            print(f"   📊 {feature}: mean={df[feature].mean():.3f}, std={df[feature].std():.3f}")
            
        self.dataset = df
        return df
        
    def prepare_features_and_target(self):
        """Prepare feature matrix and target vector"""
        
        # Combine core + anti-bias features
        all_features = self.core_features + self.antibias_features
        
        # Filter available features
        available_features = [f for f in all_features if f in self.dataset.columns]
        print(f"🎯 Using {len(available_features)} features: {available_features}")
        
        # Prepare feature matrix
        X = self.dataset[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Prepare target
        y = self.dataset['binary_target'].values
        
        # Create team groups for Group K-Fold (prevent team-specific overfitting)
        team_groups = self.dataset['HomeTeam'] + '_vs_' + self.dataset['AwayTeam']
        
        print(f"📊 Feature matrix: {X.shape}")
        print(f"🎯 Target distribution: {y.mean():.1%} home wins")
        print(f"👥 Team groups: {len(team_groups.unique())} unique matchups")
        
        return X, y, team_groups, available_features
        
    def train_with_group_kfold(self, X, y, team_groups):
        """Train model with Group K-Fold validation to prevent overfitting"""
        
        print("\n🎯 Training Binary Anti-Bias Model with Group K-Fold...")
        
        # Initialize model with balanced class weights
        base_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            max_features='sqrt',
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',  # Address class imbalance
            random_state=42,
            n_jobs=-1
        )
        
        # Use calibration for better probability estimates
        self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        
        # Group K-Fold validation (Gemini's suggestion)
        gkf = GroupKFold(n_splits=5)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall', 
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Perform cross-validation
        print("🔄 Performing Group K-Fold cross-validation...")
        cv_results = cross_validate(
            self.model, X, y, groups=team_groups, 
            cv=gkf, scoring=scoring, n_jobs=-1,
            return_train_score=False
        )
        
        # Display results
        print("\n📊 Cross-Validation Results:")
        print("-" * 50)
        
        for metric in scoring.keys():
            scores = cv_results[f'test_{metric}']
            print(f"{metric.capitalize():<12}: {scores.mean():.4f} ± {scores.std():.4f}")
            
            # Check against targets
            if metric == 'f1':
                target_met = "✅" if scores.mean() >= self.target_metrics['f1_score'] else "🔴"
                print(f"              Target: {self.target_metrics['f1_score']:.3f} {target_met}")
                
        # Train final model on full data
        print("\n🎯 Training final model on full dataset...")
        self.model.fit(X, y)
        
        # Calculate log-loss on full dataset (approximate)
        y_proba = self.model.predict_proba(X)
        logloss = log_loss(y, y_proba)
        
        print(f"📊 Final Model Metrics:")
        print(f"   Log-Loss: {logloss:.4f} (target: ≤{self.target_metrics['log_loss']})")
        
        # Feature importance
        try:
            # Access feature importance from calibrated classifier
            base_clf = self.model.calibrated_classifiers_[0].estimator
            if hasattr(base_clf, 'feature_importances_'):
                self.feature_importance = base_clf.feature_importances_
        except Exception as e:
            print(f"⚠️  Could not extract feature importance: {e}")
            self.feature_importance = None
            
        return cv_results
        
    def evaluate_on_epl_2025_26(self):
        """Test model on EPL 2025-26 data to see if it handles concept drift"""
        
        print("\n🧪 Testing on EPL 2025-26 data (concept drift test)...")
        
        # Get EPL 2025-26 data (established teams only)
        promoted_teams = ['Leeds', 'Sunderland', 'Burnley']
        
        epl_2025_data = pd.read_csv(self.dataset_path)
        epl_2025_data['Date'] = pd.to_datetime(epl_2025_data['Date'])
        
        epl_2025_established = epl_2025_data[
            (epl_2025_data['Season'] == '2025-2026') &
            (~epl_2025_data['HomeTeam'].isin(promoted_teams)) &
            (~epl_2025_data['AwayTeam'].isin(promoted_teams))
        ].copy()
        
        if len(epl_2025_established) == 0:
            print("⚠️  No EPL 2025-26 established teams data available for testing")
            return None
            
        print(f"📊 EPL 2025-26 test data: {len(epl_2025_established)} matches")
        
        # Engineer features for test data
        df = epl_2025_established.copy()
        
        # Apply same feature engineering
        df['uncertainty_amplified'] = df['market_entropy_norm'] * (
            1 + abs(df['elo_diff_normalized'] - 0.5) * 2
        )
        df['elo_confidence_zone'] = np.select([
            (df['elo_diff_normalized'] >= 0.45) & (df['elo_diff_normalized'] <= 0.55),
            df['elo_diff_normalized'] < 0.45,
            df['elo_diff_normalized'] > 0.55
        ], [0, -1, 1], default=0)
        df['away_strength_composite'] = (
            df['away_xg_eff_10'] * 0.4 +
            (1 - df['elo_diff_normalized']) * 0.3 +
            (df['away_goals_sum_5'] / 10) * 0.3
        )
        df['home_vulnerability'] = (
            (1 - df['home_xg_eff_10']) * 0.5 +
            df['market_entropy_norm'] * 0.5
        )
        df['form_momentum'] = df['form_diff_normalized'] * (
            1 + abs(df['form_diff_normalized'] - 0.5)
        )
        df['historical_home_disadvantage'] = np.select([
            df['h2h_score'] < 0.4,
            df['h2h_score'] > 0.6,
        ], [1, -1], default=0)
        
        # Prepare features
        all_features = self.core_features + self.antibias_features
        available_features = [f for f in all_features if f in df.columns]
        
        X_test = df[available_features].fillna(df[available_features].median())
        y_test = (df['FullTimeResult'] == 'H').astype(int)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logloss = log_loss(y_test, y_proba)
        
        print(f"📊 EPL 2025-26 Performance:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f} (target: ≥{self.target_metrics['precision']:.3f})")
        print(f"   Recall:    {recall:.4f} (target: ≥{self.target_metrics['recall']:.3f})")
        print(f"   F1-Score:  {f1:.4f} (target: ≥{self.target_metrics['f1_score']:.3f})")
        print(f"   Log-Loss:  {logloss:.4f} (target: ≤{self.target_metrics['log_loss']:.3f})")
        
        # Target achievement
        targets_met = {
            'precision': precision >= self.target_metrics['precision'],
            'recall': recall >= self.target_metrics['recall'],
            'f1_score': f1 >= self.target_metrics['f1_score'],
            'log_loss': logloss <= self.target_metrics['log_loss']
        }
        
        targets_achieved = sum(targets_met.values())
        print(f"\n🎯 Targets achieved: {targets_achieved}/4")
        
        for metric, achieved in targets_met.items():
            status = "✅" if achieved else "🔴"
            print(f"   {status} {metric}")
            
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Home Win', 'Home Win']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'log_loss': logloss,
            'targets_met': targets_met,
            'targets_achieved': targets_achieved,
            'test_matches': len(epl_2025_established)
        }
        
    def generate_feature_importance_analysis(self, feature_names):
        """Analyze and visualize feature importance"""
        
        if self.feature_importance is None:
            print("⚠️  Feature importance not available")
            return None
            
        print("\n📊 Feature Importance Analysis:")
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top Features:")
        for _, row in importance_df.head(10).iterrows():
            bar_length = int(row['importance'] * 50)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"   {row['feature']:<25} {bar} {row['importance']:.4f}")
            
        # Identify anti-bias feature performance
        antibias_importance = importance_df[
            importance_df['feature'].isin(self.antibias_features)
        ]
        
        if len(antibias_importance) > 0:
            avg_antibias = antibias_importance['importance'].mean()
            avg_core = importance_df[
                importance_df['feature'].isin(self.core_features)
            ]['importance'].mean()
            
            print(f"\n🛠️ Feature Type Analysis:")
            print(f"   Anti-bias features avg: {avg_antibias:.4f}")
            print(f"   Core features avg: {avg_core:.4f}")
            print(f"   Anti-bias effectiveness: {'✅ HIGH' if avg_antibias > avg_core else '🟡 MODERATE'}")
            
        return importance_df
        
    def save_model_and_report(self, cv_results, epl_test_results, importance_df):
        """Save trained model and comprehensive report"""
        
        print("\n💾 Saving binary anti-bias model and report...")
        
        # Create output directories
        models_dir = Path("models/binary_antibias")
        reports_dir = Path("results/binary_antibias")
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_file = models_dir / f"binary_antibias_model_{timestamp}.joblib"
        joblib.dump({
            'model': self.model,
            'features': self.core_features + self.antibias_features,
            'target_metrics': self.target_metrics,
            'feature_importance': importance_df.to_dict('records') if importance_df is not None else None
        }, model_file)
        
        # Compile comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'type': 'Binary Anti-Bias RandomForest + Calibration',
                'target': 'Home Win (1) vs Not Home Win (0)',
                'features_used': len(self.core_features + self.antibias_features),
                'core_features': self.core_features,
                'antibias_features': self.antibias_features,
                'training_samples': len(self.dataset)
            },
            'cross_validation_results': {
                metric: {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'scores': scores.tolist()
                }
                for metric, scores in cv_results.items() 
                if metric.startswith('test_')
            },
            'epl_2025_26_test': epl_test_results,
            'target_metrics': self.target_metrics,
            'feature_importance': importance_df.to_dict('records') if importance_df is not None else None,
            'model_path': str(model_file),
            'recommendations': self.generate_model_recommendations(cv_results, epl_test_results)
        }
        
        # Save report
        report_file = reports_dir / f"binary_antibias_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"✅ Model saved: {model_file}")
        print(f"✅ Report saved: {report_file}")
        
        return model_file, report_file
        
    def generate_model_recommendations(self, cv_results, epl_test_results):
        """Generate recommendations based on model performance"""
        
        recommendations = []
        
        # Cross-validation performance
        f1_score = cv_results['test_f1'].mean()
        
        if f1_score >= self.target_metrics['f1_score']:
            recommendations.append({
                'type': 'model_performance',
                'status': 'success',
                'message': f'F1-Score target achieved ({f1_score:.3f} ≥ {self.target_metrics["f1_score"]:.3f})',
                'action': 'Model ready for cascade integration as Stage 1'
            })
        else:
            recommendations.append({
                'type': 'model_performance',
                'status': 'improvement_needed',
                'message': f'F1-Score below target ({f1_score:.3f} < {self.target_metrics["f1_score"]:.3f})',
                'action': 'Consider hyperparameter tuning or additional feature engineering'
            })
            
        # EPL 2025-26 performance
        if epl_test_results and epl_test_results['targets_achieved'] >= 3:
            recommendations.append({
                'type': 'concept_drift_handling',
                'status': 'success',
                'message': f'Good performance on EPL 2025-26 data ({epl_test_results["targets_achieved"]}/4 targets)',
                'action': 'Model handles concept drift well, ready for production'
            })
        elif epl_test_results:
            recommendations.append({
                'type': 'concept_drift_handling',
                'status': 'warning',
                'message': f'Moderate performance on EPL 2025-26 data ({epl_test_results["targets_achieved"]}/4 targets)',
                'action': 'Consider domain adaptation or additional 2025-26 training data'
            })
            
        return recommendations
        
    def run_complete_analysis(self):
        """Run complete binary anti-bias model development and analysis"""
        
        print("🎯 BINARY ANTI-BIAS MODEL DEVELOPMENT")
        print("="*50)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Engineer anti-bias features
        self.engineer_antibias_features()
        
        # Prepare features and target
        X, y, team_groups, feature_names = self.prepare_features_and_target()
        
        # Train with Group K-Fold
        cv_results = self.train_with_group_kfold(X, y, team_groups)
        
        # Test on EPL 2025-26
        epl_test_results = self.evaluate_on_epl_2025_26()
        
        # Feature importance analysis
        importance_df = self.generate_feature_importance_analysis(feature_names)
        
        # Save model and report
        model_file, report_file = self.save_model_and_report(cv_results, epl_test_results, importance_df)
        
        print(f"\n🎉 Binary Anti-Bias Model Development Completed!")
        print(f"📊 Check detailed results in: {report_file}")
        
        return {
            'model_file': model_file,
            'report_file': report_file,
            'cv_results': cv_results,
            'epl_test_results': epl_test_results,
            'feature_importance': importance_df
        }

def main():
    """Main execution function"""
    
    # Configuration
    dataset_path = "data/processed/v15_final_enhanced.csv"
    
    # Initialize model
    model_dev = BinaryAntiBiasModel(dataset_path)
    
    # Run complete analysis
    results = model_dev.run_complete_analysis()
    
    return model_dev, results

if __name__ == "__main__":
    model_dev, results = main()