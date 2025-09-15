#!/usr/bin/env python3
"""
Improved ROI Simulator - With calibration and conservative strategy
Addresses overconfidence and betting strategy issues identified in diagnostic.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import logging
from datetime import datetime
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedROISimulator:
    """Improved ROI simulator with calibration and conservative strategy."""
    
    def __init__(self, calibration_method='isotonic'):
        """
        Args:
            calibration_method: 'isotonic' or 'sigmoid' for probability calibration
        """
        self.calibration_method = calibration_method
        
        # MUCH more conservative parameters
        self.kelly_fraction = 0.05      # 5% of Kelly (ultra-conservative)  
        self.min_edge = 0.02           # 2% minimum edge (lower bar)
        self.max_bet_size = 0.005      # Max 0.5% of bankroll per bet
        self.max_odds = 4.0            # Don't bet on long-shots
        self.min_odds = 1.4            # Focus on more predictable bets
        
        self.betting_history = []
        self.calibrated_model = None
    
    def load_and_prepare_data(self):
        """Load historical data with proper preprocessing."""
        
        logger.info("Loading historical data...")
        
        # Load odds data
        odds_files = glob.glob('/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/*.csv')
        all_odds = []
        for file in sorted(odds_files):
            season_data = pd.read_csv(file)
            season = file.split('_')[-2] + '_' + file.split('_')[-1].replace('.csv', '')
            season_data['Season'] = season
            all_odds.append(season_data)
        
        odds_df = pd.concat(all_odds, ignore_index=True)
        odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Load features
        features_df = pd.read_csv('data/processed/v13_xg_corrected_features_latest.csv')
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        
        # Merge
        merged_df = features_df.merge(
            odds_df[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A']],
            on=['Date', 'HomeTeam', 'AwayTeam'],
            how='inner'
        )
        
        logger.info(f"Loaded {len(merged_df)} matches with features and odds")
        return merged_df
    
    def train_calibrated_model(self, df):
        """Train model with proper calibration."""
        
        # Best features
        features = [
            'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
            'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
            'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
        ]
        
        available_features = [f for f in features if f in df.columns]
        df_clean = df.dropna(subset=available_features + ['B365H', 'B365D', 'B365A'])
        
        # Split data: train/val/test
        n_total = len(df_clean)
        train_end = int(n_total * 0.6)
        val_end = int(n_total * 0.8)
        
        df_train = df_clean[:train_end]
        df_val = df_clean[train_end:val_end]  
        df_test = df_clean[val_end:]
        
        # Prepare training data
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        X_train = df_train[available_features]
        y_train = df_train['FullTimeResult'].map(target_mapping)
        X_val = df_val[available_features]
        y_val = df_val['FullTimeResult'].map(target_mapping)
        
        # Train base model
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info(f"Training base model on {len(X_train)} matches...")
        base_model.fit(X_train, y_train)
        
        # Calibrate model using validation set
        logger.info("Calibrating model probabilities...")
        self.calibrated_model = CalibratedClassifierCV(
            base_model, 
            method=self.calibration_method,
            cv='prefit'  # Use prefit model
        )
        
        self.calibrated_model.fit(X_val, y_val)
        
        logger.info("Model training and calibration complete")
        
        return df_test.reset_index(drop=True), available_features
    
    def simulate_conservative_betting(self, df_test, features):
        """Run improved betting simulation with calibration."""
        
        initial_bankroll = 10000
        current_bankroll = initial_bankroll
        
        total_bets = 0
        winning_bets = 0
        total_staked = 0
        
        logger.info("Starting improved betting simulation...")
        
        # Get calibrated predictions
        X_test = df_test[features]
        calibrated_probs = self.calibrated_model.predict_proba(X_test)
        
        for i, (_, row) in enumerate(df_test.iterrows()):
            if i >= len(calibrated_probs):
                break
            
            # Calibrated model probabilities
            model_prob_home, model_prob_draw, model_prob_away = calibrated_probs[i]
            
            # Real odds
            odds_home = row['B365H']
            odds_draw = row['B365D']
            odds_away = row['B365A']
            
            if pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away):
                continue
            
            # Market probabilities (normalized)
            prob_home_implied = 1 / odds_home
            prob_draw_implied = 1 / odds_draw  
            prob_away_implied = 1 / odds_away
            
            total_implied = prob_home_implied + prob_draw_implied + prob_away_implied
            prob_home_market = prob_home_implied / total_implied
            prob_draw_market = prob_draw_implied / total_implied
            prob_away_market = prob_away_implied / total_implied
            
            # Conservative bet selection
            bets = [
                ('H', model_prob_home, prob_home_market, odds_home),
                ('D', model_prob_draw, prob_draw_market, odds_draw),
                ('A', model_prob_away, prob_away_market, odds_away)
            ]
            
            best_bet = None
            best_edge = 0
            
            for outcome, model_p, market_p, odds in bets:
                edge = model_p - market_p
                
                # MUCH more conservative selection criteria
                if (edge > self.min_edge and 
                    self.min_odds <= odds <= self.max_odds and
                    edge > best_edge and
                    model_p > 0.25):  # Don't bet on anything model thinks < 25%
                    
                    best_bet = (outcome, model_p, market_p, odds, edge)
                    best_edge = edge
            
            # Place bet if criteria met
            if best_bet is not None:
                outcome, model_p, market_p, odds, edge = best_bet
                
                # Ultra-conservative Kelly sizing
                if odds > 1:
                    b = odds - 1
                    # Add safety margin to Kelly calculation
                    adjusted_model_p = model_p * 0.9  # Discount model confidence by 10%
                    kelly_f = (b * adjusted_model_p - (1 - adjusted_model_p)) / b
                    
                    if kelly_f > 0:  # Only bet if Kelly is positive
                        bet_size_fraction = max(0, min(kelly_f * self.kelly_fraction, self.max_bet_size))
                        bet_amount = current_bankroll * bet_size_fraction
                        
                        if bet_amount >= 20:  # Minimum £20 bet
                            total_bets += 1
                            total_staked += bet_amount
                            
                            # Check result
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
                                'outcome': outcome,
                                'bet_amount': bet_amount,
                                'odds': odds,
                                'model_prob': model_p,
                                'market_prob': market_p,
                                'edge': edge,
                                'won': won,
                                'profit': profit,
                                'bankroll': current_bankroll
                            })
                            
                            # Safety stop
                            if current_bankroll < initial_bankroll * 0.5:
                                logger.warning("Bankroll dropped below 50%, stopping")
                                break
        
        # Calculate results
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
        """Generate improved performance report."""
        
        print("\n" + "="*80)
        print("🎯 ODDSY IMPROVED ROI SIMULATION")
        print("="*80)
        
        print(f"\n💰 FINANCIAL PERFORMANCE:")
        print(f"   • Initial Bankroll: £{results['initial_bankroll']:,}")
        print(f"   • Final Bankroll: £{results['final_bankroll']:,.0f}")
        print(f"   • Total Profit/Loss: £{results['total_profit']:,.0f}")
        print(f"   • ROI: {results['roi_percent']:+.2f}%")
        print(f"   • Yield per Bet: {results['yield_percent']:+.2f}%")
        
        print(f"\n📊 BETTING STATISTICS:")
        print(f"   • Total Bets: {results['total_bets']}")
        print(f"   • Winning Bets: {results['winning_bets']}")
        print(f"   • Hit Rate: {results['hit_rate']:.1%}")
        print(f"   • Total Staked: £{results['total_staked']:,.0f}")
        print(f"   • Average Edge: {results['avg_edge']:.1%}")
        print(f"   • Average Odds: {results['avg_odds']:.2f}")
        
        # Performance vs original
        print(f"\n📈 IMPROVEMENTS:")
        print(f"   • Strategy: Calibrated + Conservative")
        print(f"   • Odds Range: {self.min_odds}-{self.max_odds} (vs unlimited)")
        print(f"   • Min Edge: {self.min_edge:.1%} (vs 5%)")
        print(f"   • Kelly Fraction: {self.kelly_fraction:.1%} (vs 10%)")
        
        # Assessment
        if results['roi_percent'] > 10:
            print("\n🚀 EXCELLENT: Strong improvement achieved!")
        elif results['roi_percent'] > 0:
            print("\n✅ POSITIVE: Successfully turned profitable!")
        elif results['roi_percent'] > -2:
            print("\n⚡ IMPROVED: Much better than -5% baseline!")
        else:
            print("\n📊 STILL NEGATIVE: Further improvements needed")
        
        # Sample bets
        if len(self.betting_history) > 0:
            print(f"\n🎲 SAMPLE BETS:")
            for bet in self.betting_history[:5]:
                status = "✅ WIN" if bet['won'] else "❌ LOSS"
                print(f"   • {bet['match']}: {bet['outcome']} @ {bet['odds']:.2f} "
                      f"(Edge: {bet['edge']:+.1%}) → {status} £{bet['profit']:+.0f}")

def run_improved_simulation():
    """Run improved ROI simulation."""
    
    logger.info("🚀 Starting IMPROVED ROI simulation...")
    
    try:
        simulator = ImprovedROISimulator(calibration_method='isotonic')
        
        # Load data
        df = simulator.load_and_prepare_data()
        
        # Train calibrated model
        df_test, features = simulator.train_calibrated_model(df)
        
        if len(df_test) < 50:
            logger.error("Insufficient test data")
            return None
        
        # Run simulation
        results = simulator.simulate_conservative_betting(df_test, features)
        
        # Generate report
        simulator.generate_report(results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        results_path = f'evaluation/reports/improved_roi_simulation_{timestamp}.json'
        
        import json
        with open(results_path, 'w') as f:
            json_results = {k: float(v) if isinstance(v, np.floating) else 
                           int(v) if isinstance(v, np.integer) else v 
                           for k, v in results.items()}
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"✅ Improved simulation complete! Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Improved simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_improved_simulation()