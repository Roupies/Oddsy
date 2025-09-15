#!/usr/bin/env python3
"""
Real Odds ROI Simulator - Using actual historical bookmaker odds
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
from datetime import datetime
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealOddsROISimulator:
    """ROI simulation using real bookmaker odds from football-data.co.uk"""
    
    def __init__(self, kelly_fraction=0.1, min_edge=0.05, max_bet_size=0.02):
        """
        Args:
            kelly_fraction: Conservative Kelly scaling (0.1 = 10% of Kelly)
            min_edge: Minimum edge to place bet (5%)
            max_bet_size: Max fraction of bankroll per bet (2%)
        """
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bet_size = max_bet_size
        
        self.betting_history = []
    
    def load_historical_data(self):
        """Load all historical odds data and match with our processed features."""
        
        logger.info("Loading historical odds data...")
        
        # Load all odds files
        odds_files = glob.glob('/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/*.csv')
        
        all_odds = []
        for file in sorted(odds_files):
            season_data = pd.read_csv(file)
            # Add season identifier
            season = file.split('_')[-2] + '_' + file.split('_')[-1].replace('.csv', '')
            season_data['Season'] = season
            all_odds.append(season_data)
        
        odds_df = pd.concat(all_odds, ignore_index=True)
        logger.info(f"Loaded {len(odds_df)} matches with odds from {len(odds_files)} seasons")
        
        # Load our processed features
        features_df = pd.read_csv('data/processed/v13_xg_corrected_features_latest.csv')
        
        # Standardize team names and dates for matching
        odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%d/%m/%Y', errors='coerce')
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        
        # Merge on date, home team, away team
        merged_df = features_df.merge(
            odds_df[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A']],
            on=['Date', 'HomeTeam', 'AwayTeam'],
            how='inner'
        )
        
        logger.info(f"Successfully matched {len(merged_df)} matches with both features and odds")
        
        return merged_df
    
    def train_model_and_predict(self, df):
        """Train model and generate predictions."""
        
        # Best performing features
        features = [
            'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
            'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
            'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
        ]
        
        # Filter available features and clean data
        available_features = [f for f in features if f in df.columns]
        df_clean = df.dropna(subset=available_features + ['B365H', 'B365D', 'B365A'])
        
        logger.info(f"Using {len(available_features)} features on {len(df_clean)} matches")
        
        # Train/test split (80/20)
        split_idx = int(len(df_clean) * 0.8)
        df_train = df_clean[:split_idx]
        df_test = df_clean[split_idx:]
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        X_train = df_train[available_features]
        y_train = df_train['FullTimeResult'].map(target_mapping)
        
        model.fit(X_train, y_train)
        
        # Generate predictions on test set
        X_test = df_test[available_features]
        model_probs = model.predict_proba(X_test)
        
        logger.info(f"Model trained on {len(X_train)} matches, testing on {len(X_test)} matches")
        
        return model, model_probs, df_test.reset_index(drop=True)
    
    def simulate_betting(self, model_probs, df_test):
        """Simulate betting using real bookmaker odds."""
        
        initial_bankroll = 10000  # £10,000 starting capital
        current_bankroll = initial_bankroll
        
        total_bets = 0
        winning_bets = 0
        total_staked = 0
        
        logger.info("Starting betting simulation with real odds...")
        
        for i, (_, row) in enumerate(df_test.iterrows()):
            if i >= len(model_probs):
                break
            
            # Model probabilities
            model_prob_home, model_prob_draw, model_prob_away = model_probs[i]
            
            # Real bookmaker odds (Bet365)
            odds_home = row['B365H']
            odds_draw = row['B365D'] 
            odds_away = row['B365A']
            
            # Skip if any odds are missing
            if pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away):
                continue
            
            # Convert odds to implied probabilities
            prob_home_implied = 1 / odds_home
            prob_draw_implied = 1 / odds_draw
            prob_away_implied = 1 / odds_away
            
            # Normalize to remove bookmaker margin
            total_implied = prob_home_implied + prob_draw_implied + prob_away_implied
            prob_home_market = prob_home_implied / total_implied
            prob_draw_market = prob_draw_implied / total_implied  
            prob_away_market = prob_away_implied / total_implied
            
            # Check for value bets
            bets_to_evaluate = [
                ('H', model_prob_home, prob_home_market, odds_home),
                ('D', model_prob_draw, prob_draw_market, odds_draw), 
                ('A', model_prob_away, prob_away_market, odds_away)
            ]
            
            best_bet = None
            best_edge = 0
            
            for outcome, model_p, market_p, odds in bets_to_evaluate:
                edge = model_p - market_p
                
                # Must have significant edge and reasonable odds
                if edge > self.min_edge and edge > best_edge and 1.3 <= odds <= 15:
                    best_bet = (outcome, model_p, market_p, odds, edge)
                    best_edge = edge
            
            # Place bet if value found
            if best_bet is not None:
                outcome, model_p, market_p, odds, edge = best_bet
                
                # Kelly bet sizing
                if odds > 1:
                    b = odds - 1
                    kelly_f = (b * model_p - (1 - model_p)) / b
                    
                    # Conservative scaling with limits
                    bet_size_fraction = max(0, min(kelly_f * self.kelly_fraction, self.max_bet_size))
                    bet_amount = current_bankroll * bet_size_fraction
                    
                    if bet_amount >= 10:  # Minimum £10 bet
                        total_bets += 1
                        total_staked += bet_amount
                        
                        # Check actual result
                        actual_result = row['FullTimeResult']
                        won = (actual_result == outcome)
                        
                        if won:
                            winning_bets += 1
                            profit = bet_amount * (odds - 1)
                            current_bankroll += profit
                        else:
                            current_bankroll -= bet_amount
                            profit = -bet_amount
                        
                        # Record bet
                        self.betting_history.append({
                            'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                            'date': row['Date'],
                            'bet_outcome': outcome,
                            'bet_amount': bet_amount,
                            'odds': odds,
                            'model_prob': model_p,
                            'market_prob': market_p,
                            'edge': edge,
                            'won': won,
                            'profit': profit,
                            'bankroll_after': current_bankroll
                        })
                        
                        # Stop if bankroll drops too low
                        if current_bankroll < initial_bankroll * 0.3:
                            logger.warning("Bankroll dropped below 30% of initial, stopping simulation")
                            break
        
        # Calculate final metrics
        final_profit = current_bankroll - initial_bankroll
        roi_percent = (final_profit / initial_bankroll) * 100
        hit_rate = winning_bets / total_bets if total_bets > 0 else 0
        yield_percent = final_profit / total_staked * 100 if total_staked > 0 else 0
        
        return {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': current_bankroll,
            'total_profit': final_profit,
            'roi_percent': roi_percent,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'hit_rate': hit_rate,
            'total_staked': total_staked,
            'yield_percent': yield_percent,
            'avg_edge': np.mean([bet['edge'] for bet in self.betting_history]) if self.betting_history else 0,
            'avg_odds': np.mean([bet['odds'] for bet in self.betting_history]) if self.betting_history else 0
        }
    
    def generate_report(self, results):
        """Generate comprehensive ROI report."""
        
        print("\n" + "="*80)
        print("🎯 ODDSY REAL ODDS ROI SIMULATION")
        print("="*80)
        
        print(f"\n💰 FINANCIAL PERFORMANCE:")
        print(f"   • Initial Bankroll: £{results['initial_bankroll']:,}")
        print(f"   • Final Bankroll: £{results['final_bankroll']:,.0f}")
        print(f"   • Total Profit/Loss: £{results['total_profit']:,.0f}")
        print(f"   • ROI: {results['roi_percent']:+.2f}%")
        print(f"   • Yield per Bet: {results['yield_percent']:+.2f}%")
        
        print(f"\n📊 BETTING STATISTICS:")
        print(f"   • Total Bets Placed: {results['total_bets']}")
        print(f"   • Winning Bets: {results['winning_bets']}")
        print(f"   • Hit Rate: {results['hit_rate']:.1%}")
        print(f"   • Total Amount Staked: £{results['total_staked']:,.0f}")
        print(f"   • Average Edge Found: {results['avg_edge']:.1%}")
        print(f"   • Average Odds: {results['avg_odds']:.2f}")
        
        # Performance assessment with realistic benchmarks
        if results['roi_percent'] > 20:
            print("\n🚀 EXCEPTIONAL: Outstanding performance!")
        elif results['roi_percent'] > 10:
            print("\n✅ EXCELLENT: Strong profitability!")
        elif results['roi_percent'] > 5:
            print("\n⚡ GOOD: Solid returns!")
        elif results['roi_percent'] > 0:
            print("\n💡 POSITIVE: Beating the market!")
        else:
            print("\n📊 NEGATIVE: Needs improvement")
        
        # Show sample bets
        if len(self.betting_history) > 0:
            print(f"\n🎲 SAMPLE BETS:")
            sample_size = min(5, len(self.betting_history))
            for bet in self.betting_history[:sample_size]:
                status = "✅ WIN" if bet['won'] else "❌ LOSS"
                print(f"   • {bet['match']}: {bet['bet_outcome']} @ {bet['odds']:.2f} "
                      f"(Edge: {bet['edge']:+.1%}) → {status} £{bet['profit']:+.0f}")
        
        return results

def run_real_odds_simulation():
    """Run complete simulation using real historical bookmaker odds."""
    
    logger.info("🚀 Starting REAL odds ROI simulation...")
    
    try:
        # Initialize simulator
        simulator = RealOddsROISimulator(
            kelly_fraction=0.1,   # 10% of Kelly (conservative)
            min_edge=0.05,        # 5% minimum edge required
            max_bet_size=0.02     # Max 2% of bankroll per bet
        )
        
        # Load data with real odds
        merged_data = simulator.load_historical_data()
        
        if len(merged_data) < 100:
            logger.error("Insufficient matched data for simulation")
            return None
        
        # Train model and get predictions
        model, model_probs, df_test = simulator.train_model_and_predict(merged_data)
        
        # Run betting simulation
        results = simulator.simulate_betting(model_probs, df_test)
        
        # Generate report
        simulator.generate_report(results)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        results_path = f'evaluation/reports/real_odds_roi_simulation_{timestamp}.json'
        
        import json
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {k: float(v) if isinstance(v, np.floating) else 
                           int(v) if isinstance(v, np.integer) else v 
                           for k, v in results.items()}
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"✅ Real odds simulation complete! Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Real odds simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_real_odds_simulation()