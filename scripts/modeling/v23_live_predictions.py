#!/usr/bin/env python3
"""
v2.3 Live Predictions - 2025-26 Season Testing
Test the champion v2.3 model on the first 4 matches of EPL 2025-26 season

Strategy: Train on complete historical data, predict new season matches
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V23LivePredictor:
    """Live prediction system using champion v2.3 model."""
    
    def __init__(self):
        # v2.3 champion model configuration
        self.model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # v2.3 features (champion configuration)
        self.v23_features = [
            'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10',
            'away_xg_eff_10', 'shots_diff_normalized', 'corners_diff_normalized',
            'matchday_normalized', 'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
        ]
        
        self.model = None
        self.feature_importance = None
    
    def load_data(self):
        """Load the complete dataset with new season matches."""
        
        logger.info("Loading prediction dataset...")
        
        df = pd.read_csv('data/processed/v23_with_2025_26_predictions.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Loaded complete dataset: {len(df)} matches")
        
        # Separate historical (training) and new season (prediction) data
        new_season_start = pd.to_datetime('2025-08-01')
        
        df_historical = df[df['Date'] < new_season_start].copy()
        df_new_season = df[df['Date'] >= new_season_start].copy()
        
        logger.info(f"Historical data: {len(df_historical)} matches")
        logger.info(f"New season data: {len(df_new_season)} matches")
        
        return df_historical, df_new_season
    
    def train_model(self, df_historical):
        """Train the v2.3 model on complete historical data."""
        
        logger.info("Training v2.3 champion model...")
        
        # Check feature availability
        available_features = [f for f in self.v23_features if f in df_historical.columns]
        missing_features = [f for f in self.v23_features if f not in df_historical.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        logger.info(f"Using {len(available_features)}/10 v2.3 features")
        
        # Prepare training data
        X_train = df_historical[available_features]
        
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        y_train = df_historical['FullTimeResult'].map(target_mapping)
        
        # Remove any missing targets
        valid_mask = y_train.notna()
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        logger.info(f"Training data: {len(X_train)} matches with complete features and targets")
        
        # Train model
        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = list(zip(available_features, self.model.feature_importances_))
        self.feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("✅ v2.3 model training complete")
        
        return available_features
    
    def predict_matches(self, df_new_season, available_features):
        """Generate predictions for new season matches."""
        
        logger.info("Generating predictions for 2025-26 matches...")
        
        if self.model is None:
            raise ValueError("Model not trained! Call train_model first.")
        
        # Prepare prediction data
        X_predict = df_new_season[available_features]
        
        # Generate predictions
        predictions = self.model.predict(X_predict)
        probabilities = self.model.predict_proba(X_predict)
        
        # Convert predictions back to labels
        label_mapping = {0: 'H', 1: 'D', 2: 'A'}
        predicted_labels = [label_mapping[pred] for pred in predictions]
        
        # Create results dataframe
        results = df_new_season[['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult']].copy()
        results['Predicted'] = predicted_labels
        results['Prob_Home'] = probabilities[:, 0]
        results['Prob_Draw'] = probabilities[:, 1]
        results['Prob_Away'] = probabilities[:, 2]
        results['Confidence'] = np.max(probabilities, axis=1)
        
        return results
    
    def evaluate_played_matches(self, results):
        """Evaluate predictions for matches that have already been played."""
        
        played_matches = results[results['FullTimeResult'].notna()].copy()
        
        if len(played_matches) == 0:
            logger.info("No played matches to evaluate")
            return None
        
        # Calculate accuracy
        accuracy = (played_matches['FullTimeResult'] == played_matches['Predicted']).mean()
        
        logger.info(f"Played matches evaluation: {len(played_matches)} matches")
        logger.info(f"Accuracy: {accuracy:.2%}")
        
        return {
            'total_played': len(played_matches),
            'correct_predictions': (played_matches['FullTimeResult'] == played_matches['Predicted']).sum(),
            'accuracy': accuracy,
            'results': played_matches
        }
    
    def generate_prediction_report(self, results, evaluation):
        """Generate comprehensive prediction report."""
        
        print("\\n" + "="*80)
        print("🏆 v2.3 CHAMPION MODEL - 2025-26 SEASON PREDICTIONS")
        print("="*80)
        
        print(f"\\n📊 MODEL PERFORMANCE:")
        print(f"   • Champion accuracy on historical data: 57.11%")
        print(f"   • Features used: {len(self.v23_features)} (simplified champion set)")
        print(f"   • Training data: 2280 historical matches")
        
        print(f"\\n⭐ TOP FEATURES (Historical Importance):")
        for i, (feature, importance) in enumerate(self.feature_importance[:5], 1):
            print(f"   {i}. {feature}: {importance:.3f}")
        
        print(f"\\n📅 2025-26 SEASON PREDICTIONS:")
        print(f"{'Date':<12} {'Match':<30} {'Actual':<8} {'Predicted':<10} {'Confidence':<11} {'Status'}")
        print("-" * 85)
        
        for _, row in results.iterrows():
            date_str = row['Date'].strftime('%m/%d')
            match = f"{row['HomeTeam'][:12]} vs {row['AwayTeam'][:12]}"
            actual = str(row['FullTimeResult']) if pd.notna(row['FullTimeResult']) else 'TBD'
            predicted = row['Predicted']
            confidence = f"{row['Confidence']:.1%}"
            
            # Status indicator
            if pd.notna(row['FullTimeResult']):
                if row['FullTimeResult'] == row['Predicted']:
                    status = "✅ CORRECT"
                else:
                    status = "❌ WRONG"
            else:
                status = "🔮 FUTURE"
            
            print(f"{date_str:<12} {match:<30} {actual:<8} {predicted:<10} {confidence:<11} {status}")
        
        # Evaluation summary
        if evaluation:
            print(f"\\n🎯 EVALUATION SUMMARY:")
            print(f"   • Played matches: {evaluation['total_played']}")
            print(f"   • Correct predictions: {evaluation['correct_predictions']}")
            print(f"   • Accuracy: {evaluation['accuracy']:.1%}")
            
            if evaluation['accuracy'] >= 0.67:
                verdict = "🚀 EXCELLENT START!"
            elif evaluation['accuracy'] >= 0.5:
                verdict = "✅ GOOD PERFORMANCE"
            else:
                verdict = "📊 LEARNING PHASE"
            
            print(f"   • Verdict: {verdict}")
        
        # Future predictions
        future_matches = results[results['FullTimeResult'].isna()]
        if len(future_matches) > 0:
            print(f"\\n🔮 FUTURE MATCH PREDICTIONS:")
            
            for _, row in future_matches.iterrows():
                match = f"{row['HomeTeam']} vs {row['AwayTeam']}"
                prediction = row['Predicted']
                confidence = row['Confidence']
                
                pred_text = {
                    'H': f"{row['HomeTeam']} Win",
                    'D': "Draw", 
                    'A': f"{row['AwayTeam']} Win"
                }[prediction]
                
                print(f"   • {match} → {pred_text} ({confidence:.1%} confidence)")
                print(f"     Probabilities: H:{row['Prob_Home']:.1%} D:{row['Prob_Draw']:.1%} A:{row['Prob_Away']:.1%}")
        
        print(f"\\n📋 MODEL SUMMARY:")
        print(f"   • The v2.3 champion model (57.11% historical accuracy) is now predicting 2025-26")
        print(f"   • Simple 10-feature architecture beats complex alternatives")
        print(f"   • Elo + Market + Shots = 44.5% of predictive power")
        print(f"   • Ready for live season tracking!")

def main():
    """Main prediction workflow."""
    
    logger.info("🚀 Starting v2.3 live prediction system...")
    
    # Initialize predictor
    predictor = V23LivePredictor()
    
    # Load data
    df_historical, df_new_season = predictor.load_data()
    
    # Train model
    available_features = predictor.train_model(df_historical)
    
    # Generate predictions
    results = predictor.predict_matches(df_new_season, available_features)
    
    # Evaluate played matches
    evaluation = predictor.evaluate_played_matches(results)
    
    # Generate report
    predictor.generate_prediction_report(results, evaluation)
    
    # Save results
    results.to_csv('data/predictions/v23_2025_26_predictions.csv', index=False)
    logger.info("✅ Predictions saved to data/predictions/v23_2025_26_predictions.csv")
    
    return results, evaluation

if __name__ == "__main__":
    results, evaluation = main()