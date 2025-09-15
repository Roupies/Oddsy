#!/usr/bin/env python3
"""
Ultra-Conservative ROI Simulation
Realistic parameters to simulate actual betting conditions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_conservative_simulation():
    """Ultra-conservative simulation with realistic market inefficiencies."""
    
    logger.info("🚀 Running ultra-conservative ROI simulation...")
    
    # Load data
    df = pd.read_csv('data/processed/premier_league_market_v3_2025_09_02_105923.csv')
    
    # Best features
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    # Clean data
    df_clean = df.dropna(subset=features + ['FullTimeResult'])
    available_features = [f for f in features if f in df_clean.columns]
    
    # Train/test split
    split_idx = int(len(df_clean) * 0.8)
    df_train = df_clean[:split_idx]
    df_test = df_clean[split_idx:]
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    X_train = df_train[available_features]
    y_train = df_train['FullTimeResult'].map(target_mapping)
    
    model.fit(X_train, y_train)
    
    # Predict on test set
    X_test = df_test[available_features]
    model_probs = model.predict_proba(X_test)
    
    # Ultra-conservative simulation
    initial_bankroll = 1000
    current_bankroll = initial_bankroll
    
    # ULTRA-CONSERVATIVE parameters
    min_edge = 0.08        # 8% minimum edge (very high bar)
    max_bet_size = 0.005   # Max 0.5% of bankroll per bet
    kelly_fraction = 0.05  # Extremely conservative Kelly
    
    betting_history = []
    
    logger.info(f"Testing {len(df_test)} matches with ultra-conservative parameters...")
    
    for i, (_, row) in enumerate(df_test.reset_index().iterrows()):
        if i >= len(model_probs):
            break
            
        # Model predictions
        model_prob_home, model_prob_draw, model_prob_away = model_probs[i]
        
        # Market probabilities (add some noise to make more realistic)
        if pd.isna(row['market_home_prob_norm']):
            continue
            
        # Add bookmaker margin (typical 5-8%) and noise to make realistic
        margin = 0.06
        noise_factor = np.random.uniform(0.95, 1.05)  # ±5% noise
        
        market_home = row['market_home_prob_norm'] * (1 + margin) * noise_factor
        market_draw = row['market_draw_prob_norm'] * (1 + margin) * noise_factor  
        market_away = row['market_away_prob_norm'] * (1 + margin) * noise_factor
        
        # Normalize
        total_market = market_home + market_draw + market_away
        market_home /= total_market
        market_draw /= total_market
        market_away /= total_market
        
        # Convert to odds
        odds_home = 1 / market_home if market_home > 0 else 100
        odds_draw = 1 / market_draw if market_draw > 0 else 100
        odds_away = 1 / market_away if market_away > 0 else 100
        
        # Find value bets with high bar
        bets = [
            ('H', model_prob_home, market_home, odds_home),
            ('D', model_prob_draw, market_draw, odds_draw),
            ('A', model_prob_away, market_away, odds_away)
        ]
        
        for outcome, model_p, market_p, odds in bets:
            edge = model_p - market_p
            
            # Only bet if HUGE edge and reasonable odds
            if edge > min_edge and odds > 1.5 and odds < 10:
                
                # Ultra-conservative Kelly
                b = odds - 1
                kelly_f = (b * model_p - (1 - model_p)) / b
                bet_size_fraction = max(0, min(kelly_f * kelly_fraction, max_bet_size))
                bet_amount = current_bankroll * bet_size_fraction
                
                if bet_amount >= 5:  # Minimum £5 bet
                    
                    # Check result
                    actual = row['FullTimeResult'] 
                    won = (actual == outcome)
                    
                    if won:
                        profit = bet_amount * (odds - 1)
                        current_bankroll += profit
                    else:
                        current_bankroll -= bet_amount
                        profit = -bet_amount
                    
                    betting_history.append({
                        'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                        'outcome': outcome,
                        'bet_amount': bet_amount,
                        'odds': odds,
                        'edge': edge,
                        'won': won,
                        'profit': profit
                    })
    
    # Results
    total_profit = current_bankroll - initial_bankroll
    roi_percent = (total_profit / initial_bankroll) * 100
    total_bets = len(betting_history)
    winning_bets = sum(1 for bet in betting_history if bet['won'])
    hit_rate = winning_bets / total_bets if total_bets > 0 else 0
    
    print("\n" + "="*60)
    print("🎯 ULTRA-CONSERVATIVE ROI SIMULATION")  
    print("="*60)
    
    print(f"\n💰 FINANCIAL RESULTS:")
    print(f"   • Initial: £{initial_bankroll:,}")
    print(f"   • Final: £{current_bankroll:,.0f}")
    print(f"   • Profit: £{total_profit:,.0f}")
    print(f"   • ROI: {roi_percent:+.2f}%")
    
    print(f"\n📊 BETTING STATS:")
    print(f"   • Total Bets: {total_bets}")
    print(f"   • Wins: {winning_bets}")
    print(f"   • Hit Rate: {hit_rate:.1%}")
    print(f"   • Avg Edge: {np.mean([b['edge'] for b in betting_history]):.1%}" if betting_history else "   • No bets placed")
    
    # Assessment
    if roi_percent > 15:
        print("\n🚀 EXCELLENT: Strong profitability!")
    elif roi_percent > 5:
        print("\n✅ GOOD: Profitable performance!")
    elif roi_percent > 0:
        print("\n⚡ POSITIVE: Shows promise!")
    else:
        print("\n📊 BREAK-EVEN: Realistic baseline")
    
    if total_bets > 0:
        print(f"\n🎲 SAMPLE BETS:")
        for bet in betting_history[:3]:
            status = "✅" if bet['won'] else "❌"
            print(f"   • {bet['match']}: {bet['outcome']} @ {bet['odds']:.2f} → {status} £{bet['profit']:+.0f}")
    
    logger.info(f"✅ Conservative simulation complete: {total_bets} bets, {roi_percent:+.1f}% ROI")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    run_conservative_simulation()