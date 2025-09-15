#!/usr/bin/env python3
"""
Rolling EPL 2025-26 Validation Script

Tests the v2.3 production model in rolling validation on EPL 2025-26:
- Predict J1 → integrate real results → predict J2 → etc.
- Track performance on 3 promoted teams (Leeds, Sunderland, Burnley)
- Generate comprehensive metrics and insights

Usage:
    python rolling_epl_2025_26_validator.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our team initialization module
from team_initialization import TeamInitializer

# Configuration
CONFIG = {
    'model_path': 'models/v23_retrained_2025_09_11_154613.joblib',
    'dataset_path': 'data/processed/v15_final_enhanced.csv',
    'calendar_path': 'data/raw/epl-2025-2026_GMTStandardTime.csv',
    'promoted_teams': ['Leeds', 'Sunderland', 'Burnley'],
    'production_features': [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized', 
        'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ],
    'gameweeks_to_validate': [1, 2, 3, 4],  # Already played gameweeks
    'output_dir': 'results/rolling_validation_2025_26'
}

class EPLRollingValidator:
    """
    Rolling validation system for EPL 2025-26 season
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.historical_data = None
        self.calendar_data = None
        self.promoted_teams = config['promoted_teams']
        self.features = config['production_features']
        
        # Initialize team state manager
        self.team_initializer = TeamInitializer()
        self.team_states = {}
        
        # Results tracking
        self.results = {
            'predictions': [],
            'actuals': [],
            'gameweek_accuracy': {},
            'cumulative_accuracy': {},
            'promoted_analysis': {},
            'class_breakdown': {'H': [], 'D': [], 'A': []},
            'feature_states': {}
        }
        
        # Create output directory
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load model, historical data, and calendar"""
        print("🔄 Loading model and data...")
        
        # Load production model
        self.model = joblib.load(self.config['model_path'])
        print(f"✅ Loaded model: {self.config['model_path']}")
        
        # Load historical dataset (v15)
        self.historical_data = pd.read_csv(self.config['dataset_path'])
        self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
        print(f"✅ Loaded historical data: {len(self.historical_data)} matches")
        
        # Load EPL 2025-26 calendar
        self.calendar_data = pd.read_csv(self.config['calendar_path'])
        self.calendar_data['Date'] = pd.to_datetime(self.calendar_data['Date'], format='%d/%m/%Y %H:%M')
        
        # Clean team names for consistency
        self.calendar_data = self._clean_team_names(self.calendar_data)
        print(f"✅ Loaded EPL 2025-26 calendar: {len(self.calendar_data)} matches")
        
        # Initialize all team states for 2025-26
        print("\n🏟️  Setting up team initial states...")
        self.team_states = self.team_initializer.initialize_all_teams(self.historical_data)
        
    def _clean_team_names(self, df):
        """Standardize team names between datasets"""
        team_mapping = {
            'Man Utd': 'Man United',
            'Spurs': 'Tottenham', 
            "Nott'm Forest": 'Nott\'m Forest',
            'Brighton': 'Brighton',
            'Wolves': 'Wolves'
        }
        
        for old_name, new_name in team_mapping.items():
            df['Home Team'] = df['Home Team'].str.replace(old_name, new_name)
            df['Away Team'] = df['Away Team'].str.replace(old_name, new_name)
            
        return df
        
    def _parse_result(self, result_str):
        """Parse result string '4 - 2' into outcome H/D/A"""
        if pd.isna(result_str) or result_str == '':
            return None
            
        try:
            home_goals, away_goals = map(int, result_str.split(' - '))
            if home_goals > away_goals:
                return 'H'
            elif home_goals < away_goals:
                return 'A'
            else:
                return 'D'
        except:
            return None
            
    def extract_gameweek_matches(self, gameweek):
        """Extract matches for specific gameweek"""
        gw_matches = self.calendar_data[
            self.calendar_data['Round Number'] == gameweek
        ].copy()
        
        # Parse actual results
        gw_matches['Actual_Result'] = gw_matches['Result'].apply(self._parse_result)
        
        return gw_matches
        
    def get_latest_features_state(self, until_date):
        """Get latest feature values for all teams up to specific date"""
        # Filter historical data up to the date
        historical_until = self.historical_data[
            self.historical_data['Date'] <= until_date
        ].copy()
        
        if len(historical_until) == 0:
            # Return neutral features if no historical data
            return self._get_neutral_features()
            
        # Get latest values for each team
        team_features = {}
        
        # Extract unique teams from latest season
        latest_season_data = historical_until[
            historical_until['Season'] == historical_until['Season'].max()
        ]
        
        all_teams = set(list(latest_season_data['HomeTeam'].unique()) + 
                       list(latest_season_data['AwayTeam'].unique()))
        
        # For each team, get their latest feature values
        for team in all_teams:
            team_matches = latest_season_data[
                (latest_season_data['HomeTeam'] == team) |
                (latest_season_data['AwayTeam'] == team)
            ].sort_values('Date')
            
            if len(team_matches) > 0:
                # Get features from most recent match
                latest_match = team_matches.iloc[-1]
                team_features[team] = {
                    'elo_base': self._extract_elo_for_team(latest_match, team),
                    'form_base': self._extract_form_for_team(latest_match, team),
                    'xg_eff': self._extract_xg_eff_for_team(latest_match, team)
                }
        
        return team_features
        
    def _get_neutral_features(self):
        """Return neutral feature values for initialization"""
        return {
            'elo_diff_normalized': 0.5,
            'market_entropy_norm': 0.5,
            'shots_diff_normalized': 0.5,
            'corners_diff_normalized': 0.5,
            'form_diff_normalized': 0.5,
            'h2h_score': 0.5,
            'matchday_normalized': 0.0,
            'home_xg_eff_10': 0.5,
            'away_xg_eff_10': 0.5,
            'away_goals_sum_5': 0.5
        }
        
    def _extract_elo_for_team(self, match_row, team):
        """Extract Elo-related info for specific team"""
        is_home = (match_row['HomeTeam'] == team)
        # Simplified extraction - in real implementation would be more complex
        return 0.5  # Neutral value for now
        
    def _extract_form_for_team(self, match_row, team):
        """Extract form info for specific team"""
        return 0.5  # Neutral value for now
        
    def _extract_xg_eff_for_team(self, match_row, team):
        """Extract xG efficiency for specific team"""
        return 0.5  # Neutral value for now
        
    def create_prediction_features(self, match_row, gameweek):
        """Create feature vector for prediction using team initialization"""
        home_team = match_row['Home Team']
        away_team = match_row['Away Team']
        
        # Use team initializer to create proper features
        features, match_info = self.team_initializer.create_match_features(
            home_team, away_team, self.team_states, gameweek
        )
        
        # Add match info to the match row for analysis
        for key, value in match_info.items():
            match_row[key] = value
        
        return features
        
    def predict_gameweek(self, gameweek):
        """Predict all matches in a gameweek"""
        print(f"\n🎯 Predicting Gameweek {gameweek}...")
        
        # Get matches for this gameweek
        gw_matches = self.extract_gameweek_matches(gameweek)
        
        if len(gw_matches) == 0:
            print(f"⚠️  No matches found for gameweek {gameweek}")
            return
            
        predictions = []
        actuals = []
        
        for idx, match in gw_matches.iterrows():
            if match['Actual_Result'] is None:
                continue
                
            # Create features for prediction
            features = self.create_prediction_features(match, gameweek)
            
            # Make prediction
            feature_vector = np.array([[features[f] for f in self.features]])
            pred_proba = self.model.predict_proba(feature_vector)[0]
            pred_class = self.model.predict(feature_vector)[0]
            
            # Convert prediction class to H/D/A
            class_mapping = {0: 'H', 1: 'D', 2: 'A'}
            pred_outcome = class_mapping[pred_class]
            
            # Store prediction and actual
            prediction_record = {
                'gameweek': gameweek,
                'match_id': match['Match Number'],
                'date': match['Date'],
                'home_team': match['Home Team'],
                'away_team': match['Away Team'],
                'actual_result': match['Actual_Result'],
                'predicted_result': pred_outcome,
                'predicted_probas': {
                    'H': float(pred_proba[0]),
                    'D': float(pred_proba[1]),
                    'A': float(pred_proba[2])
                },
                'confidence': float(max(pred_proba)),
                'correct': pred_outcome == match['Actual_Result'],
                'involves_promoted': match.get('involves_promoted', False),
                'promoted_teams': [t for t in [match['Home Team'], match['Away Team']] 
                                 if t in self.promoted_teams]
            }
            
            predictions.append(prediction_record)
            actuals.append(match['Actual_Result'])
            
        # Store results
        self.results['predictions'].extend(predictions)
        self.results['actuals'].extend(actuals)
        
        # Calculate gameweek accuracy
        correct_predictions = sum(1 for p in predictions if p['correct'])
        gw_accuracy = correct_predictions / len(predictions) if len(predictions) > 0 else 0
        self.results['gameweek_accuracy'][gameweek] = {
            'accuracy': gw_accuracy,
            'correct': correct_predictions,
            'total': len(predictions),
            'matches': predictions
        }
        
        print(f"✅ Gameweek {gameweek}: {correct_predictions}/{len(predictions)} "
              f"({gw_accuracy:.1%} accuracy)")
        
        return predictions
        
    def calculate_cumulative_metrics(self):
        """Calculate cumulative accuracy metrics"""
        print("\n📊 Calculating cumulative metrics...")
        
        all_predictions = self.results['predictions']
        
        if len(all_predictions) == 0:
            print("⚠️  No predictions to analyze")
            return
            
        # Overall accuracy
        total_correct = sum(1 for p in all_predictions if p['correct'])
        overall_accuracy = total_correct / len(all_predictions)
        
        print(f"🎯 Overall Accuracy: {total_correct}/{len(all_predictions)} "
              f"({overall_accuracy:.1%})")
        
        # Class breakdown
        class_stats = {'H': {'correct': 0, 'total': 0},
                      'D': {'correct': 0, 'total': 0},
                      'A': {'correct': 0, 'total': 0}}
        
        for pred in all_predictions:
            actual = pred['actual_result']
            class_stats[actual]['total'] += 1
            if pred['correct']:
                class_stats[actual]['correct'] += 1
        
        print("\n📈 Class Breakdown:")
        for class_name, stats in class_stats.items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                print(f"   {class_name}: {stats['correct']}/{stats['total']} ({acc:.1%})")
        
        # Promoted teams analysis
        promoted_predictions = [p for p in all_predictions if p['involves_promoted']]
        non_promoted_predictions = [p for p in all_predictions if not p['involves_promoted']]
        
        if len(promoted_predictions) > 0:
            promoted_correct = sum(1 for p in promoted_predictions if p['correct'])
            promoted_acc = promoted_correct / len(promoted_predictions)
            print(f"\n🆕 Promoted Teams: {promoted_correct}/{len(promoted_predictions)} "
                  f"({promoted_acc:.1%})")
            
        if len(non_promoted_predictions) > 0:
            established_correct = sum(1 for p in non_promoted_predictions if p['correct'])
            established_acc = established_correct / len(non_promoted_predictions)
            print(f"🏛️  Established Teams: {established_correct}/{len(non_promoted_predictions)} "
                  f"({established_acc:.1%})")
        
        # Store cumulative results
        self.results['cumulative_accuracy'] = {
            'overall': overall_accuracy,
            'total_correct': total_correct,
            'total_predictions': len(all_predictions),
            'class_breakdown': class_stats,
            'promoted_accuracy': promoted_correct / len(promoted_predictions) if promoted_predictions else 0,
            'established_accuracy': established_correct / len(non_promoted_predictions) if non_promoted_predictions else 0
        }
        
    def generate_detailed_report(self):
        """Generate comprehensive validation report"""
        print("\n📋 Generating detailed report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_info': {
                'model': self.config['model_path'],
                'dataset': self.config['dataset_path'],
                'gameweeks_validated': self.config['gameweeks_to_validate'],
                'promoted_teams': self.promoted_teams
            },
            'summary': self.results['cumulative_accuracy'],
            'gameweek_breakdown': self.results['gameweek_accuracy'],
            'all_predictions': self.results['predictions'],
            'insights': self._generate_insights()
        }
        
        # Save report
        output_file = Path(self.config['output_dir']) / f'rolling_validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"✅ Report saved: {output_file}")
        return report
        
    def _generate_insights(self):
        """Generate key insights from validation"""
        all_preds = self.results['predictions']
        
        if len(all_preds) == 0:
            return {"error": "No predictions available for analysis"}
            
        insights = {}
        
        # Promoted teams specific analysis
        promoted_matches = [p for p in all_preds if p['involves_promoted']]
        
        for team in self.promoted_teams:
            team_matches = [p for p in all_preds if team in p['promoted_teams']]
            if team_matches:
                team_correct = sum(1 for p in team_matches if p['correct'])
                insights[f'{team}_performance'] = {
                    'accuracy': team_correct / len(team_matches),
                    'matches': len(team_matches),
                    'correct': team_correct
                }
        
        # Draw prediction analysis (known weakness)
        draw_predictions = [p for p in all_preds if p['predicted_result'] == 'D']
        actual_draws = [p for p in all_preds if p['actual_result'] == 'D']
        
        insights['draw_analysis'] = {
            'predicted_draws': len(draw_predictions),
            'actual_draws': len(actual_draws),
            'draw_recall': sum(1 for p in actual_draws if p['predicted_result'] == 'D') / len(actual_draws) if actual_draws else 0,
            'draw_precision': sum(1 for p in draw_predictions if p['correct']) / len(draw_predictions) if draw_predictions else 0
        }
        
        return insights
        
    def run_validation(self):
        """Run complete rolling validation"""
        print("🚀 Starting EPL 2025-26 Rolling Validation")
        print("="*50)
        
        # Load data
        self.load_data()
        
        # Validate each gameweek
        for gameweek in self.config['gameweeks_to_validate']:
            self.predict_gameweek(gameweek)
            
        # Calculate metrics
        self.calculate_cumulative_metrics()
        
        # Generate report
        report = self.generate_detailed_report()
        
        print("\n🎉 Rolling validation completed!")
        print("="*50)
        
        return report

def main():
    """Main execution"""
    validator = EPLRollingValidator(CONFIG)
    report = validator.run_validation()
    
    # Print key results
    summary = report['summary']
    print(f"\n🏆 FINAL RESULTS:")
    print(f"Overall Accuracy: {summary['overall']:.1%}")
    print(f"Promoted Teams: {summary.get('promoted_accuracy', 0):.1%}")
    print(f"Established Teams: {summary.get('established_accuracy', 0):.1%}")

if __name__ == "__main__":
    main()