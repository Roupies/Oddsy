#!/usr/bin/env python3
"""
Realistic ROI Simulator - Using actual model predictions and market probabilities
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealisticROISimulator:
    """ROI simulation with real model predictions and market data."""
    
    def __init__(self, kelly_fraction=0.1, min_edge=0.03, max_bet_size=0.01):
        """Conservative parameters for realistic simulation."""
        self.kelly_fraction = kelly_fraction  # Very conservative
        self.min_edge = min_edge  # 3% minimum edge
        self.max_bet_size = max_bet_size  # Max 1% of bankroll
        
        self.betting_history = []
        
    def train_and_predict(self, df):
        """Train model and generate realistic predictions."""
        
        # Use best v2.9 features
        features = [
            'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
            'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
            'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
        ]
        
        # Check available features
        available_features = [f for f in features if f in df.columns]
        logger.info(f"Using {len(available_features)} features: {available_features}")
        
        if len(available_features) < 5:
            logger.error("Insufficient features available")
            return None
        
        # Prepare data
        df_clean = df.dropna(subset=available_features + ['FullTimeResult'])
        X = df_clean[available_features]
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        y = df_clean['FullTimeResult'].map(target_mapping)
        
        # Train/test split (80/20)
        split_idx = int(len(df_clean) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
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
        
        model.fit(X_train, y_train)
        
        # Generate predictions
        model_probs = model.predict_proba(X_test)
        
        logger.info(f"Model trained on {len(X_train)} matches, predicting {len(X_test)} matches")
        
        return model_probs, df_test.reset_index(drop=True)
    
    def simulate_betting(self, model_probs, df_test):
        """Run realistic betting simulation."""
        
        initial_bankroll = 1000  # £1000 starting capital
        current_bankroll = initial_bankroll
        
        total_bets = 0
        winning_bets = 0
        total_staked = 0
        total_return = 0
        
        for i, (_, row) in enumerate(df_test.iterrows()):
            if i >= len(model_probs):
                break
            
            # Model probabilities
            model_prob_home, model_prob_draw, model_prob_away = model_probs[i]
            
            # Market probabilities (already normalized)
            if pd.isna(row['market_home_prob_norm']):
                continue
                
            market_prob_home = row['market_home_prob_norm']
            market_prob_draw = row['market_draw_prob_norm']
            market_prob_away = row['market_away_prob_norm']
            
            # Convert market probabilities to odds
            odds_home = 1 / market_prob_home if market_prob_home > 0 else 100
            odds_draw = 1 / market_prob_draw if market_prob_draw > 0 else 100
            odds_away = 1 / market_prob_away if market_prob_away > 0 else 100
            
            # Find value bets
            outcomes = [
                ('H', model_prob_home, market_prob_home, odds_home),
                ('D', model_prob_draw, market_prob_draw, odds_draw),
                ('A', model_prob_away, market_prob_away, odds_away)
            ]
            
            best_bet = None
            best_edge = 0
            
            for outcome, model_p, market_p, odds in outcomes:
                edge = model_p - market_p
                
                if edge > self.min_edge and edge > best_edge:
                    best_bet = (outcome, model_p, market_p, odds, edge)
                    best_edge = edge
            
            # Place bet if value found
            if best_bet is not None:
                outcome, model_p, market_p, odds, edge = best_bet
                
                # Kelly bet sizing
                if odds > 1 and model_p > 0:
                    # Kelly formula: f = (bp - q) / b where b = odds-1, p = model_prob, q = 1-model_prob
                    b = odds - 1
                    kelly_f = (b * model_p - (1 - model_p)) / b
                    
                    # Conservative scaling
                    bet_size_fraction = max(0, min(kelly_f * self.kelly_fraction, self.max_bet_size))
                    bet_amount = current_bankroll * bet_size_fraction
                    
                    if bet_amount >= 1:  # Minimum £1 bet
                        total_bets += 1
                        total_staked += bet_amount
                        
                        # Check result
                        actual_result = row['FullTimeResult']
                        won = (actual_result == outcome)
                        
                        if won:
                            winning_bets += 1
                            payout = bet_amount * odds
                            profit = payout - bet_amount
                            current_bankroll += profit
                            total_return += payout
                        else:
                            current_bankroll -= bet_amount
                            total_return += 0
                        
                        # Record bet
                        self.betting_history.append({
                            'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                            'date': row.get('Date', ''),
                            'bet_outcome': outcome,
                            'bet_amount': bet_amount,
                            'odds': odds,
                            'model_prob': model_p,
                            'market_prob': market_p,
                            'edge': edge,
                            'won': won,
                            'profit': profit if won else -bet_amount,
                            'bankroll_after': current_bankroll
                        })
        
        # Calculate final metrics
        total_profit = current_bankroll - initial_bankroll
        roi_percent = (total_profit / initial_bankroll) * 100
        hit_rate = winning_bets / total_bets if total_bets > 0 else 0
        yield_percent = total_profit / total_staked * 100 if total_staked > 0 else 0
        
        return {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': current_bankroll,
            'total_profit': total_profit,
            'roi_percent': roi_percent,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'hit_rate': hit_rate,
            'total_staked': total_staked,
            'yield_percent': yield_percent,
            'avg_edge': np.mean([bet['edge'] for bet in self.betting_history]) if self.betting_history else 0
        }
    
    def generate_report(self, results):
        """Generate realistic performance report."""
        
        print("\n" + "="*70)
        print("🎯 ODDSY REALISTIC ROI SIMULATION")
        print("="*70)
        
        print(f"\n💰 FINANCIAL PERFORMANCE:")
        print(f"   • Initial Bankroll: £{results['initial_bankroll']:,.0f}")
        print(f"   • Final Bankroll: £{results['final_bankroll']:,.0f}")
        print(f"   • Total Profit: £{results['total_profit']:,.0f}")
        print(f"   • ROI: {results['roi_percent']:+.2f}%")
        print(f"   • Yield per Bet: {results['yield_percent']:+.2f}%")
        
        print(f"\n📊 BETTING STATISTICS:")
        print(f"   • Total Bets: {results['total_bets']}")
        print(f"   • Winning Bets: {results['winning_bets']}")
        print(f"   • Hit Rate: {results['hit_rate']:.3f} ({results['hit_rate']*100:.1f}%)")
        print(f"   • Total Staked: £{results['total_staked']:,.0f}")
        print(f"   • Average Edge: {results['avg_edge']:.4f} ({results['avg_edge']*100:.2f}%)")
        
        # Performance benchmarks
        if results['roi_percent'] > 20:
            print("\n🚀 EXCEPTIONAL: Outstanding model performance!")
        elif results['roi_percent'] > 10:
            print("\n✅ EXCELLENT: Strong profitability achieved!")
        elif results['roi_percent'] > 5:
            print("\n⚡ GOOD: Solid edge over market!")
        elif results['roi_percent'] > 0:
            print("\n💡 POSITIVE: Model shows promise!")
        else:
            print("\n📊 NEGATIVE: Needs improvement")
        
        # Show some example bets
        if len(self.betting_history) > 0:
            print(f"\n🎲 SAMPLE BETS:")
            for bet in self.betting_history[:5]:
                status = "✅ WIN" if bet['won'] else "❌ LOSS"
                print(f"   • {bet['match']}: {bet['bet_outcome']} @ {bet['odds']:.2f} "
                      f"(Edge: {bet['edge']:.3f}) → {status} £{bet['profit']:+.0f}")
        
        return results

def run_realistic_simulation():
    """Run complete realistic ROI simulation."""
    
    logger.info("🚀 Starting realistic ROI simulation...")
    
    # Load market data
    market_data_path = 'data/processed/premier_league_market_v3_2025_09_02_105923.csv'
    
    try:
        df = pd.read_csv(market_data_path)
        logger.info(f"Loaded {len(df)} matches with market data")
        
        # Initialize simulator
        simulator = RealisticROISimulator(
            kelly_fraction=0.1,   # Very conservative
            min_edge=0.03,        # 3% minimum edge  
            max_bet_size=0.01     # Max 1% of bankroll per bet
        )
        
        # Train model and generate predictions
        model_probs, df_test = simulator.train_and_predict(df)
        
        if model_probs is None:
            logger.error("Model training failed")
            return None
        
        # Run betting simulation
        results = simulator.simulate_betting(model_probs, df_test)
        
        # Generate report
        simulator.generate_report(results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        results_path = f'evaluation/reports/realistic_roi_simulation_{timestamp}.json'
        
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Realistic simulation complete! Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        return None

if __name__ == "__main__":
    results = run_realistic_simulation()