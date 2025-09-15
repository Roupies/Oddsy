#!/usr/bin/env python3
"""
ROI Simulator - Transform Oddsy from accuracy to profitability
Simulates betting strategy using model probabilities vs bookmaker odds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValueBettingSimulator:
    """Simulates betting strategy based on model probabilities vs market odds."""
    
    def __init__(self, kelly_fraction=0.25, min_edge=0.05, max_bet_size=0.02):
        """
        Args:
            kelly_fraction: Fraction of Kelly criterion to use (conservative)
            min_edge: Minimum edge required to place bet
            max_bet_size: Maximum bet size as fraction of bankroll
        """
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bet_size = max_bet_size
        
        self.betting_history = []
        self.bankroll_history = []
        
    def calculate_implied_probabilities(self, odds_home, odds_draw, odds_away):
        """Calculate bookmaker's implied probabilities from odds."""
        
        if pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away):
            return None, None, None
        
        prob_home = 1 / odds_home
        prob_draw = 1 / odds_draw  
        prob_away = 1 / odds_away
        
        # Normalize to remove bookmaker margin
        total = prob_home + prob_draw + prob_away
        
        return prob_home/total, prob_draw/total, prob_away/total
    
    def calculate_edge(self, model_prob, market_prob):
        """Calculate edge: how much our model disagrees with market."""
        if market_prob <= 0:
            return 0
        return model_prob - market_prob
    
    def kelly_bet_size(self, edge, odds):
        """Calculate optimal bet size using Kelly criterion."""
        if edge <= 0 or odds <= 1:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds-1, p = model_prob, q = 1-model_prob
        b = odds - 1
        p = edge + (1/odds)  # Reconstruct model prob
        q = 1 - p
        
        if p <= 0 or b <= 0:
            return 0
            
        kelly_fraction = (b * p - q) / b
        
        # Apply conservative scaling and limits
        bet_size = kelly_fraction * self.kelly_fraction
        return min(bet_size, self.max_bet_size)
    
    def simulate_betting(self, df, model_predictions):
        """Run complete betting simulation."""
        
        initial_bankroll = 1000  # Start with £1000
        current_bankroll = initial_bankroll
        
        total_bets = 0
        winning_bets = 0
        total_staked = 0
        total_profit = 0
        
        logger.info("Starting betting simulation...")
        
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= len(model_predictions):
                break
                
            # Get model probabilities (assumed to be [prob_home, prob_draw, prob_away])
            if len(model_predictions[i]) != 3:
                continue
                
            model_probs = model_predictions[i]
            actual_result = row['FullTimeResult']
            
            # Get bookmaker odds (need to be in the dataset)
            if 'odds_home' not in row or 'odds_draw' not in row or 'odds_away' not in row:
                continue
                
            odds = [row['odds_home'], row['odds_draw'], row['odds_away']]
            
            # Skip if any odds are missing
            if any(pd.isna(odds)):
                continue
            
            # Calculate implied probabilities
            market_probs = self.calculate_implied_probabilities(odds[0], odds[1], odds[2])
            if market_probs[0] is None:
                continue
            
            # Check for value bets
            outcomes = ['H', 'D', 'A']
            best_bet = None
            best_edge = 0
            best_odds = 0
            
            for j, outcome in enumerate(outcomes):
                edge = self.calculate_edge(model_probs[j], market_probs[j])
                
                if edge > self.min_edge and edge > best_edge:
                    best_bet = outcome
                    best_edge = edge
                    best_odds = odds[j]
            
            # Place bet if value found
            if best_bet is not None:
                bet_size_fraction = self.kelly_bet_size(best_edge, best_odds)
                bet_amount = current_bankroll * bet_size_fraction
                
                if bet_amount > 1:  # Minimum £1 bet
                    total_bets += 1
                    total_staked += bet_amount
                    
                    # Check if bet won
                    won = (actual_result == best_bet)
                    
                    if won:
                        winning_bets += 1
                        profit = bet_amount * (best_odds - 1)
                        current_bankroll += profit
                        total_profit += profit
                    else:
                        current_bankroll -= bet_amount
                        total_profit -= bet_amount
                    
                    # Record bet
                    bet_record = {
                        'match_date': row.get('Date', i),
                        'home_team': row.get('HomeTeam', 'Unknown'),
                        'away_team': row.get('AwayTeam', 'Unknown'),
                        'bet_outcome': best_bet,
                        'bet_amount': bet_amount,
                        'odds': best_odds,
                        'edge': best_edge,
                        'won': won,
                        'profit': profit if won else -bet_amount,
                        'bankroll': current_bankroll
                    }
                    
                    self.betting_history.append(bet_record)
                    self.bankroll_history.append(current_bankroll)
        
        # Calculate final metrics
        roi = (current_bankroll - initial_bankroll) / initial_bankroll * 100
        hit_rate = winning_bets / total_bets if total_bets > 0 else 0
        avg_odds = np.mean([bet['odds'] for bet in self.betting_history]) if self.betting_history else 0
        yield_per_bet = total_profit / total_staked * 100 if total_staked > 0 else 0
        
        results = {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': current_bankroll,
            'total_profit': total_profit,
            'roi_percent': roi,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'hit_rate': hit_rate,
            'total_staked': total_staked,
            'yield_percent': yield_per_bet,
            'avg_odds': avg_odds
        }
        
        return results
    
    def generate_report(self, results):
        """Generate comprehensive betting report."""
        
        print("\n" + "="*60)
        print("🎯 ODDSY BETTING SIMULATION RESULTS")
        print("="*60)
        
        print(f"\n💰 FINANCIAL PERFORMANCE:")
        print(f"   • Initial Bankroll: £{results['initial_bankroll']:,.0f}")
        print(f"   • Final Bankroll: £{results['final_bankroll']:,.0f}")
        print(f"   • Total Profit: £{results['total_profit']:,.0f}")
        print(f"   • ROI: {results['roi_percent']:+.2f}%")
        print(f"   • Yield per Bet: {results['yield_percent']:+.2f}%")
        
        print(f"\n📊 BETTING STATISTICS:")
        print(f"   • Total Bets Placed: {results['total_bets']}")
        print(f"   • Winning Bets: {results['winning_bets']}")
        print(f"   • Hit Rate: {results['hit_rate']:.3f} ({results['hit_rate']*100:.1f}%)")
        print(f"   • Total Amount Staked: £{results['total_staked']:,.0f}")
        print(f"   • Average Odds: {results['avg_odds']:.2f}")
        
        # Performance assessment
        if results['roi_percent'] > 10:
            print("\n🚀 EXCELLENT: Model shows strong profitability!")
        elif results['roi_percent'] > 5:
            print("\n✅ GOOD: Model beats market consistently")
        elif results['roi_percent'] > 0:
            print("\n⚡ POSITIVE: Model has edge over bookmakers")
        else:
            print("\n❌ NEGATIVE: Model needs improvement")
        
        return results

def load_model_predictions(model_path, test_data_path):
    """Load model and generate predictions for ROI simulation."""
    
    # This is a placeholder - you'll need to load your actual model
    # For now, we'll simulate with dummy probabilities
    
    df = pd.read_csv(test_data_path)
    n_matches = len(df)
    
    # Simulate model probabilities (replace with your actual model)
    np.random.seed(42)
    predictions = []
    
    for i in range(n_matches):
        # Generate realistic-looking probabilities
        probs = np.random.dirichlet([2, 1, 1.5])  # Slightly favor home
        predictions.append(probs)
    
    logger.info(f"Generated predictions for {n_matches} matches")
    return predictions

def run_roi_simulation():
    """Run complete ROI simulation."""
    
    logger.info("🚀 Starting ROI simulation...")
    
    # Load test data (you'll need odds columns)
    test_data_path = 'data/processed/v211_draw_features_2025_09_06.csv'
    
    try:
        df = pd.read_csv(test_data_path)
        logger.info(f"Loaded {len(df)} matches for simulation")
        
        # For simulation, add dummy odds columns if not present
        if 'odds_home' not in df.columns:
            logger.warning("Adding dummy odds for simulation")
            np.random.seed(42)
            df['odds_home'] = np.random.uniform(1.5, 4.0, len(df))
            df['odds_draw'] = np.random.uniform(2.8, 5.0, len(df))
            df['odds_away'] = np.random.uniform(1.8, 6.0, len(df))
        
        # Load model predictions
        model_predictions = load_model_predictions(None, test_data_path)
        
        # Run simulation
        simulator = ValueBettingSimulator(
            kelly_fraction=0.25,  # Conservative Kelly
            min_edge=0.05,        # 5% minimum edge
            max_bet_size=0.02     # Max 2% of bankroll per bet
        )
        
        results = simulator.simulate_betting(df, model_predictions)
        
        # Generate report
        simulator.generate_report(results)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        results_path = f'evaluation/reports/roi_simulation_{timestamp}.json'
        
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ ROI simulation complete! Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        return None

if __name__ == "__main__":
    results = run_roi_simulation()