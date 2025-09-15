#!/usr/bin/env python3
"""
Live Predictions Pipeline for EPL 2025-26

Generate real-time predictions for upcoming EPL matches using validated v2.3 model
with lessons learned from rolling validation on promoted teams.

Usage:
    python live_predictions_pipeline.py --gameweek 5
    python live_predictions_pipeline.py --next  # Next unplayed gameweek
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from team_initialization import TeamInitializer

class LivePredictionsEngine:
    """Generate live predictions for EPL 2025-26 matches"""
    
    def __init__(self):
        self.model_path = 'models/v23_retrained_2025_09_11_154613.joblib'
        self.dataset_path = 'data/processed/v15_final_enhanced.csv'
        self.calendar_path = 'data/raw/epl-2025-2026_GMTStandardTime.csv'
        
        self.production_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        # Load components
        self.model = None
        self.historical_data = None
        self.calendar_data = None
        self.team_initializer = TeamInitializer()
        self.team_states = {}
        
        # Lessons learned from rolling validation
        self.promoted_teams = ['Leeds', 'Sunderland', 'Burnley']
        self.validation_insights = {
            'promoted_performance': 0.70,  # 70% accuracy on promoted teams
            'established_performance': 0.43,  # 43% accuracy on established teams
            'draw_weakness': True,  # 0% draw recall
            'confidence_adjustment': -0.024  # Slightly less confident on promoted teams
        }
        
    def load_components(self):
        """Load all required components"""
        print("🔄 Loading prediction components...")
        
        # Load model
        self.model = joblib.load(self.model_path)
        print(f"✅ Model loaded: {self.model_path}")
        
        # Load historical data  
        self.historical_data = pd.read_csv(self.dataset_path)
        self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
        print(f"✅ Historical data: {len(self.historical_data)} matches")
        
        # Load calendar
        self.calendar_data = pd.read_csv(self.calendar_path)
        self.calendar_data['Date'] = pd.to_datetime(self.calendar_data['Date'], format='%d/%m/%Y %H:%M')
        self.calendar_data = self._clean_team_names(self.calendar_data)
        print(f"✅ Calendar loaded: {len(self.calendar_data)} matches")
        
        # Initialize team states
        self.team_states = self.team_initializer.initialize_all_teams(self.historical_data)
        print(f"✅ Team states initialized")
        
    def _clean_team_names(self, df):
        """Standardize team names"""
        team_mapping = {
            'Man Utd': 'Man United',
            'Spurs': 'Tottenham',
            "Nott'm Forest": 'Nott\'m Forest',
        }
        
        for old_name, new_name in team_mapping.items():
            df['Home Team'] = df['Home Team'].str.replace(old_name, new_name)
            df['Away Team'] = df['Away Team'].str.replace(old_name, new_name)
            
        return df
        
    def get_next_gameweek(self):
        """Find the next unplayed gameweek"""
        for gw in range(1, 39):
            gw_matches = self.calendar_data[self.calendar_data['Round Number'] == gw]
            unplayed = gw_matches[gw_matches['Result'].isna() | (gw_matches['Result'] == '')]
            
            if len(unplayed) > 0:
                played = len(gw_matches) - len(unplayed) 
                print(f"🎯 Next gameweek: {gw} ({played}/{len(gw_matches)} matches played)")
                return gw
                
        print("⚠️  All gameweeks appear to be completed")
        return None
        
    def get_gameweek_matches(self, gameweek):
        """Get matches for specific gameweek"""
        gw_matches = self.calendar_data[
            self.calendar_data['Round Number'] == gameweek
        ].copy()
        
        # Only return unplayed matches
        unplayed = gw_matches[gw_matches['Result'].isna() | (gw_matches['Result'] == '')]
        
        return unplayed.sort_values('Date')
        
    def predict_match(self, match_row, gameweek):
        """Generate prediction for single match"""
        home_team = match_row['Home Team']
        away_team = match_row['Away Team']
        
        # Create features using team initializer
        features, match_info = self.team_initializer.create_match_features(
            home_team, away_team, self.team_states, gameweek
        )
        
        # Make prediction
        feature_vector = np.array([[features[f] for f in self.production_features]])
        pred_proba = self.model.predict_proba(feature_vector)[0]
        pred_class = self.model.predict(feature_vector)[0]
        
        # Convert to H/D/A
        class_mapping = {0: 'H', 1: 'D', 2: 'A'}
        predicted_outcome = class_mapping[pred_class]
        
        # Apply validation insights for confidence adjustment
        base_confidence = float(max(pred_proba))
        
        # Adjust confidence based on team status
        if match_info['involves_promoted']:
            # Promoted teams: higher accuracy but slightly lower model confidence
            adjusted_confidence = base_confidence * 1.05  # Boost based on observed 70% accuracy
        else:
            # Established teams: lower observed accuracy
            adjusted_confidence = base_confidence * 0.95  # Reduce based on observed 43% accuracy
            
        # Cap confidence at reasonable bounds
        adjusted_confidence = max(0.33, min(0.95, adjusted_confidence))
        
        # Create prediction record
        prediction = {
            'match_id': int(match_row['Match Number']),
            'gameweek': gameweek,
            'date': match_row['Date'].isoformat(),
            'kickoff': match_row['Date'].strftime('%H:%M'),
            'home_team': home_team,
            'away_team': away_team,
            'venue': match_row.get('Location', 'Unknown'),
            'prediction': {
                'outcome': predicted_outcome,
                'probabilities': {
                    'home_win': float(pred_proba[0]),
                    'draw': float(pred_proba[1]),
                    'away_win': float(pred_proba[2])
                },
                'confidence': adjusted_confidence,
                'raw_confidence': base_confidence
            },
            'team_info': {
                'home_status': match_info.get('home_status', 'established'),
                'away_status': match_info.get('away_status', 'established'),
                'involves_promoted': match_info['involves_promoted'],
                'promoted_teams': [t for t in [home_team, away_team] if t in self.promoted_teams]
            },
            'model_info': {
                'version': 'v2.3_production',
                'features_used': len(self.production_features),
                'validation_accuracy': '50.0% overall (70% promoted, 43% established)'
            }
        }
        
        return prediction
        
    def predict_gameweek(self, gameweek):
        """Generate predictions for entire gameweek"""
        print(f"\n🎯 Generating predictions for Gameweek {gameweek}")
        print("="*60)
        
        # Get matches
        matches = self.get_gameweek_matches(gameweek)
        
        if len(matches) == 0:
            print(f"⚠️  No unplayed matches found for gameweek {gameweek}")
            return []
            
        print(f"📅 Found {len(matches)} matches to predict:")
        
        predictions = []
        
        for idx, match in matches.iterrows():
            prediction = self.predict_match(match, gameweek)
            predictions.append(prediction)
            
            # Display prediction
            p = prediction['prediction']
            team_info = prediction['team_info']
            
            status_emoji = "🆕" if team_info['involves_promoted'] else "🏛️"
            confidence_emoji = "🟢" if p['confidence'] > 0.6 else "🟡" if p['confidence'] > 0.4 else "🔴"
            
            print(f"\n   {status_emoji} {prediction['home_team']} vs {prediction['away_team']}")
            print(f"      📍 {prediction['venue']} | ⏰ {prediction['kickoff']}")
            print(f"      🎯 Prediction: {p['outcome']} ({confidence_emoji} {p['confidence']:.1%} confidence)")
            print(f"      📊 H: {p['probabilities']['home_win']:.2f} | D: {p['probabilities']['draw']:.2f} | A: {p['probabilities']['away_win']:.2f}")
            
            if team_info['promoted_teams']:
                print(f"      🆕 Promoted: {', '.join(team_info['promoted_teams'])}")
                
        return predictions
        
    def save_predictions(self, predictions, gameweek):
        """Save predictions to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        prediction_summary = {
            'timestamp': datetime.now().isoformat(),
            'gameweek': gameweek,
            'model': {
                'version': 'v2.3_production',
                'path': self.model_path,
                'features': self.production_features
            },
            'validation_context': self.validation_insights,
            'predictions_count': len(predictions),
            'predictions': predictions,
            'summary_stats': self._calculate_prediction_stats(predictions)
        }
        
        # Create output directory
        output_dir = Path("predictions/gameweek_predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed predictions
        detailed_file = output_dir / f"gw{gameweek}_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(prediction_summary, f, indent=2, default=str)
            
        # Save simple CSV for easy consumption
        csv_data = []
        for p in predictions:
            csv_data.append({
                'Match_ID': p['match_id'],
                'Date': p['date'][:10],
                'Kickoff': p['kickoff'],
                'Home_Team': p['home_team'],
                'Away_Team': p['away_team'],
                'Prediction': p['prediction']['outcome'],
                'Confidence': f"{p['prediction']['confidence']:.1%}",
                'Home_Win_Prob': f"{p['prediction']['probabilities']['home_win']:.2f}",
                'Draw_Prob': f"{p['prediction']['probabilities']['draw']:.2f}",
                'Away_Win_Prob': f"{p['prediction']['probabilities']['away_win']:.2f}",
                'Involves_Promoted': p['team_info']['involves_promoted'],
                'Venue': p['venue']
            })
            
        csv_df = pd.DataFrame(csv_data)
        csv_file = output_dir / f"gw{gameweek}_simple_{timestamp}.csv"
        csv_df.to_csv(csv_file, index=False)
        
        print(f"\n💾 Predictions saved:")
        print(f"   📋 Detailed: {detailed_file}")
        print(f"   📊 Simple CSV: {csv_file}")
        
        return detailed_file
        
    def _calculate_prediction_stats(self, predictions):
        """Calculate summary statistics"""
        if not predictions:
            return {}
            
        outcomes = [p['prediction']['outcome'] for p in predictions]
        confidences = [p['prediction']['confidence'] for p in predictions]
        promoted_count = sum(1 for p in predictions if p['team_info']['involves_promoted'])
        
        return {
            'total_matches': len(predictions),
            'predicted_outcomes': {
                'home_wins': outcomes.count('H'),
                'draws': outcomes.count('D'),
                'away_wins': outcomes.count('A')
            },
            'confidence_stats': {
                'average': float(np.mean(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            },
            'team_composition': {
                'matches_with_promoted_teams': promoted_count,
                'matches_established_only': len(predictions) - promoted_count
            }
        }
        
    def run_live_predictions(self, gameweek=None):
        """Main execution function"""
        print("🚀 EPL 2025-26 Live Predictions Engine")
        print("="*50)
        
        # Load components
        self.load_components()
        
        # Determine gameweek
        if gameweek is None:
            gameweek = self.get_next_gameweek()
            if gameweek is None:
                return
        else:
            print(f"🎯 Target gameweek: {gameweek}")
            
        # Generate predictions
        predictions = self.predict_gameweek(gameweek)
        
        if not predictions:
            return
            
        # Save predictions
        output_file = self.save_predictions(predictions, gameweek)
        
        # Summary
        stats = self._calculate_prediction_stats(predictions)
        print(f"\n📈 Prediction Summary:")
        print(f"   Total Matches: {stats['total_matches']}")
        print(f"   Predictions: {stats['predicted_outcomes']['home_wins']}H - " +
              f"{stats['predicted_outcomes']['draws']}D - {stats['predicted_outcomes']['away_wins']}A")
        print(f"   Avg Confidence: {stats['confidence_stats']['average']:.1%}")
        print(f"   Promoted Teams: {stats['team_composition']['matches_with_promoted_teams']} matches")
        
        print(f"\n🎉 Live predictions generated! See: {output_file}")
        
        return predictions

def main():
    parser = argparse.ArgumentParser(description='Generate live EPL predictions')
    parser.add_argument('--gameweek', '-g', type=int, help='Specific gameweek to predict')
    parser.add_argument('--next', '-n', action='store_true', help='Predict next unplayed gameweek')
    
    args = parser.parse_args()
    
    engine = LivePredictionsEngine()
    
    if args.next:
        predictions = engine.run_live_predictions()
    elif args.gameweek:
        predictions = engine.run_live_predictions(args.gameweek)
    else:
        # Default: next gameweek
        predictions = engine.run_live_predictions()
        
    return predictions

if __name__ == "__main__":
    main()