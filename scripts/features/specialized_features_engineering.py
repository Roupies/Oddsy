#!/usr/bin/env python3
"""
🎯 FEATURE ENGINEERING SPÉCIALISÉ - DÉTECTION DRAWS
================================================

Script complet pour créer 5 features spécialisées RÉELLES sur l'ensemble 
du dataset (train + test) pour optimiser la cascade de détection des draws.

Features implémentées:
1. elo_variance_recent - Variance ELO sur 8-10 matchs récents
2. team_parity_score - Score de parité parfaite (elo_diff ≈ 0.5 + market_entropy)
3. market_odds_spread - Écart-type des cotes bookmakers multiples
4. low_scoring_potential - Potentiel match faible score (xG efficiency)
5. is_promoted - Flag exact équipes promues par saison
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("specialized_features")

class SpecializedFeatureCalculator:
    """Calculateur de features spécialisées pour détection draws"""
    
    def __init__(self):
        self.promoted_teams_by_season = {
            '2019-2020': ['Sheffield United', 'Norwich', 'Aston Villa'],
            '2020-2021': ['West Brom', 'Fulham', 'Leeds'], 
            '2021-2022': ['Brentford', 'Norwich', 'Watford'],
            '2022-2023': ['Fulham', 'Bournemouth', "Nott'm Forest"],
            '2023-2024': ['Burnley', 'Sheffield United', 'Luton'],
            '2024-2025': ['Leicester', 'Ipswich', 'Southampton'],
            '2025-2026': ['Leeds', 'Sunderland', 'Burnley']
        }
        logger.info("🎯 SpecializedFeatureCalculator initialisé")
    
    def calculate_elo_variance_recent(self, df):
        """
        Feature 1: Variance des ratings ELO sur les 8-10 derniers matchs
        """
        logger.info("📊 Calcul elo_variance_recent...")
        
        df = df.copy()
        df['elo_variance_recent'] = 0.5  # Valeur par défaut
        
        # Pour chaque équipe et chaque match, calculer variance ELO récente
        teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        
        for team in teams:
            # Tous les matchs de cette équipe (domicile + extérieur)
            team_matches = df[
                (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
            ].sort_values('Date')
            
            if len(team_matches) < 10:
                continue
                
            # Pour chaque match, calculer variance sur fenêtre précédente
            for i in range(10, len(team_matches)):
                current_match_idx = team_matches.index[i]
                
                # 10 matchs précédents
                window_matches = team_matches.iloc[i-10:i]
                elo_values = []
                
                for _, match in window_matches.iterrows():
                    if match['HomeTeam'] == team:
                        # ELO domicile = base + diff
                        elo_home = 0.5 + (match['elo_diff_normalized'] - 0.5)
                        elo_values.append(elo_home)
                    else:
                        # ELO extérieur = base - diff
                        elo_away = 0.5 - (match['elo_diff_normalized'] - 0.5)
                        elo_values.append(elo_away)
                
                # Variance ELO (normalisée)
                if len(elo_values) >= 8:
                    elo_variance = np.var(elo_values)
                    # Normaliser variance (clampée 0-1)
                    elo_variance_norm = min(max(elo_variance * 10, 0), 1)
                    df.at[current_match_idx, 'elo_variance_recent'] = elo_variance_norm
        
        logger.info(f"   ✅ elo_variance_recent calculé (moyenne: {df['elo_variance_recent'].mean():.3f})")
        return df
    
    def calculate_team_parity_score(self, df):
        """
        Feature 2: Score de parité parfaite combinant elo_diff ≈ 0.5 + market_entropy
        """
        logger.info("⚖️  Calcul team_parity_score...")
        
        df = df.copy()
        
        # Composante 1: Distance à 0.5 pour elo_diff (plus proche = plus de parité)
        elo_parity_component = 1 - abs(df['elo_diff_normalized'] - 0.5) * 2
        
        # Composante 2: Market entropy (déjà normalisé 0-1)  
        market_component = df['market_entropy_norm']
        
        # Score combiné (moyenne pondérée: 70% elo, 30% market)
        df['team_parity_score'] = (0.7 * elo_parity_component + 0.3 * market_component)
        
        # S'assurer que c'est entre 0-1
        df['team_parity_score'] = df['team_parity_score'].clip(0, 1)
        
        logger.info(f"   ✅ team_parity_score calculé (moyenne: {df['team_parity_score'].mean():.3f})")
        return df
    
    def calculate_market_odds_spread(self, df, raw_data_path):
        """
        Feature 3: Écart-type des cotes bookmakers (incertitude marché)
        """
        logger.info("💰 Calcul market_odds_spread...")
        
        try:
            # Charger données market
            df_market = pd.read_csv(raw_data_path, encoding='utf-8-sig')
            
            # Nettoyer les noms d'équipes pour matching
            team_mapping = {
                'Spurs': 'Tottenham',
                "Nott'm Forest": "Nott'm Forest"
            }
            if 'HomeTeam' in df_market.columns:
                df_market['HomeTeam'] = df_market['HomeTeam'].replace(team_mapping)
                df_market['AwayTeam'] = df_market['AwayTeam'].replace(team_mapping)
            
            df = df.copy()
            df['market_odds_spread'] = 0.5  # Défaut
            
            # Colonnes de cotes disponibles (varie selon dataset)
            odds_cols_home = [col for col in df_market.columns if any(x in col for x in ['B365H', 'PSH', 'WH', 'BFDH'])]
            odds_cols_draw = [col for col in df_market.columns if any(x in col for x in ['B365D', 'PSD', 'WD', 'BFDD'])]
            odds_cols_away = [col for col in df_market.columns if any(x in col for x in ['B365A', 'PSA', 'WA', 'BFDA'])]
            
            logger.info(f"   Trouvé {len(odds_cols_home)} bookmakers pour odds Home")
            
            if len(odds_cols_home) >= 2:  # Au moins 2 bookmakers
                for i, row in df.iterrows():
                    # Matcher avec les données market
                    match_market = df_market[
                        (df_market['HomeTeam'] == row['HomeTeam']) & 
                        (df_market['AwayTeam'] == row['AwayTeam'])
                    ]
                    
                    if len(match_market) > 0:
                        market_row = match_market.iloc[0]
                        
                        # Collecter les cotes H/D/A de différents bookmakers
                        home_odds = [market_row[col] for col in odds_cols_home if pd.notna(market_row[col]) and market_row[col] > 0]
                        draw_odds = [market_row[col] for col in odds_cols_draw if pd.notna(market_row[col]) and market_row[col] > 0]
                        away_odds = [market_row[col] for col in odds_cols_away if pd.notna(market_row[col]) and market_row[col] > 0]
                        
                        # Calculer spread (écart-type moyen)
                        spreads = []
                        if len(home_odds) >= 2:
                            spreads.append(np.std(home_odds))
                        if len(draw_odds) >= 2:
                            spreads.append(np.std(draw_odds))
                        if len(away_odds) >= 2:
                            spreads.append(np.std(away_odds))
                        
                        if len(spreads) > 0:
                            # Moyenne des spreads, normalisée (divisée par 3 pour avoir ordre 0-1)
                            market_spread = np.mean(spreads) / 3
                            df.at[i, 'market_odds_spread'] = min(max(market_spread, 0), 1)
            
            logger.info(f"   ✅ market_odds_spread calculé (moyenne: {df['market_odds_spread'].mean():.3f})")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Erreur market_odds_spread: {e}, utilisation valeur défaut 0.5")
            df['market_odds_spread'] = 0.5
            
        return df
    
    def calculate_low_scoring_potential(self, df):
        """
        Feature 4: Potentiel match à faible score (via xG efficiency)
        """
        logger.info("⚽ Calcul low_scoring_potential...")
        
        df = df.copy()
        
        # Moyenne des efficacités xG des deux équipes
        # Plus faible efficacité = plus de chance de match serré/draw
        avg_xg_efficiency = (df['home_xg_eff_10'] + df['away_xg_eff_10']) / 2
        
        # Inverser: faible efficacité → fort potentiel faible score
        df['low_scoring_potential'] = 1 - avg_xg_efficiency
        
        # S'assurer entre 0-1
        df['low_scoring_potential'] = df['low_scoring_potential'].clip(0, 1)
        
        logger.info(f"   ✅ low_scoring_potential calculé (moyenne: {df['low_scoring_potential'].mean():.3f})")
        return df
    
    def calculate_promoted_flags(self, df):
        """
        Feature 5: Flags binaires équipes promues par saison
        """
        logger.info("🆙 Calcul is_promoted flags...")
        
        df = df.copy()
        df['is_promoted'] = 0
        
        for season, teams in self.promoted_teams_by_season.items():
            mask = df['Season'] == season
            for team in teams:
                promoted_mask = mask & ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))
                df.loc[promoted_mask, 'is_promoted'] = 1
        
        promoted_count = df['is_promoted'].sum()
        total_matches = len(df)
        
        logger.info(f"   ✅ is_promoted calculé: {promoted_count}/{total_matches} matchs avec équipe promue ({promoted_count/total_matches*100:.1f}%)")
        return df
    
    def engineer_all_features(self, df_path, raw_market_paths):
        """
        Orchestrateur principal: calculer les 5 features spécialisées
        """
        logger.info("🚀 DÉBUT FEATURE ENGINEERING SPÉCIALISÉ")
        logger.info("=" * 55)
        
        # Charger dataset principal
        df = pd.read_csv(df_path, parse_dates=['Date'])
        logger.info(f"📊 Dataset chargé: {len(df)} matchs, {len(df.columns)} features")
        
        # Calculer features une par une
        df = self.calculate_elo_variance_recent(df)
        df = self.calculate_team_parity_score(df)  
        
        # Essayer plusieurs sources market
        market_spread_calculated = False
        for market_path in raw_market_paths:
            try:
                logger.info(f"   Tentative market_odds_spread avec: {market_path}")
                df_test = self.calculate_market_odds_spread(df.copy(), market_path)
                if df_test['market_odds_spread'].std() > 0.01:  # Variance significative
                    df = df_test
                    market_spread_calculated = True
                    logger.info(f"   ✅ Utilisé {market_path} pour market_odds_spread")
                    break
            except Exception as e:
                logger.warning(f"   Échec {market_path}: {e}")
        
        if not market_spread_calculated:
            df['market_odds_spread'] = 0.5
            logger.warning("   ⚠️ market_odds_spread défaut (0.5)")
        
        df = self.calculate_low_scoring_potential(df)
        df = self.calculate_promoted_flags(df)
        
        logger.info(f"✅ FEATURES SPÉCIALISÉES COMPLÉTÉES")
        logger.info(f"   Total features: {len(df.columns)} (15 originales + 5 nouvelles)")
        
        # Vérifications finales
        new_features = ['elo_variance_recent', 'team_parity_score', 'market_odds_spread', 
                       'low_scoring_potential', 'is_promoted']
        
        for feature in new_features:
            missing = df[feature].isna().sum()
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            logger.info(f"   {feature}: {missing} manquants, moyenne={mean_val:.3f}, std={std_val:.3f}")
        
        return df

def main():
    """Test du feature engineering spécialisé"""
    
    try:
        # Paths
        processed_data = 'data/processed/v15_final_enhanced.csv'
        raw_market_data = [
            'data/raw/E0 (7).csv',  # EPL 2025-26 avec multiples bookmakers
            'data/raw/premier_league_2019_2024.csv'  # Historique
        ]
        output_path = 'data/processed/v16_specialized_features_enhanced.csv'
        
        # Feature engineering
        calculator = SpecializedFeatureCalculator()
        df_enhanced = calculator.engineer_all_features(processed_data, raw_market_data)
        
        # Export
        df_enhanced.to_csv(output_path, index=False)
        logger.info(f"💾 Dataset augmenté sauvé: {output_path}")
        
        # Statistiques finales
        logger.info(f"\n📈 RÉSUMÉ FINAL:")
        logger.info(f"   Matchs traités: {len(df_enhanced)}")
        logger.info(f"   Features totales: {len(df_enhanced.columns)}")
        logger.info(f"   Nouvelles features: 5 spécialisées pour draws")
        logger.info(f"   Fichier: {output_path}")
        
        return df_enhanced
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    if result is not None:
        print("✅ Feature engineering spécialisé terminé avec succès")
    else:
        print("❌ Échec du feature engineering")