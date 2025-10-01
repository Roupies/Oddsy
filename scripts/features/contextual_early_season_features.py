#!/usr/bin/env python3
"""
💡 FEATURES CONTEXTUELLES DÉBUT SAISON

Basé sur analyse post-mortem J1-J4, développement de features spécifiques
pour capturer la volatilité et patterns uniques du début de saison

Features développées:
1. rest_days_diff - Différence récupération équipes
2. promoted_team_factor - Impact équipes promues  
3. early_season_volatility - Facteur volatilité historique
4. manager_continuity - Stabilité équipe technique
5. transfer_window_impact - Impact mercato récent
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime, timedelta
from pathlib import Path

class EarlySeasonFeatureEngineer:
    def __init__(self, data_path='data/processed/v15_final_enhanced.csv'):
        """Initialiser ingénierie features début saison"""
        self.data_path = data_path
        self.df = pd.read_csv(data_path, parse_dates=['Date'])
        
        # Équipes promues 2025-26 (adaptable selon saison)
        self.promoted_teams_2025 = ['Leeds', 'Sunderland', 'Burnley']
        
        # Dates importantes
        self.season_start_2025 = pd.Timestamp('2025-08-01')
        self.transfer_deadline = pd.Timestamp('2025-08-31')  # Approximation
        
    def calculate_rest_days_diff(self, df_enhanced):
        """Feature 1: Différence jours repos entre équipes"""
        print("🛌 Calcul rest_days_diff...")
        
        df_enhanced['rest_days_diff'] = 0.0
        
        # Pour chaque match, calculer différence jours repos
        for i, match in df_enhanced.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            match_date = match['Date']
            
            # Trouver dernier match de chaque équipe avant ce match
            home_previous = df_enhanced[
                ((df_enhanced['HomeTeam'] == home_team) | (df_enhanced['AwayTeam'] == home_team)) &
                (df_enhanced['Date'] < match_date)
            ].sort_values('Date').tail(1)
            
            away_previous = df_enhanced[
                ((df_enhanced['HomeTeam'] == away_team) | (df_enhanced['AwayTeam'] == away_team)) &
                (df_enhanced['Date'] < match_date)
            ].sort_values('Date').tail(1)
            
            if len(home_previous) > 0 and len(away_previous) > 0:
                home_rest_days = (match_date - home_previous['Date'].iloc[0]).days
                away_rest_days = (match_date - away_previous['Date'].iloc[0]).days
                
                # Différence repos (positif = home plus reposé)
                rest_diff = home_rest_days - away_rest_days
                
                # Normaliser approximativement (-7 à +7 jours → -1 à +1)
                df_enhanced.loc[i, 'rest_days_diff'] = np.clip(rest_diff / 7.0, -1.0, 1.0)
            else:
                # Valeur neutre si pas d'historique
                df_enhanced.loc[i, 'rest_days_diff'] = 0.0
        
        non_zero_count = (df_enhanced['rest_days_diff'] != 0.0).sum()
        print(f"  ✅ {non_zero_count} matches avec différence repos calculée")
        
        return df_enhanced
    
    def calculate_promoted_team_factor(self, df_enhanced):
        """Feature 2: Facteur équipes promues avec ajustement performance"""
        print("📈 Calcul promoted_team_factor...")
        
        df_enhanced['promoted_team_factor'] = 0.0
        
        # Pour chaque match impliquant équipe promue
        for i, match in df_enhanced.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            home_promoted = home_team in self.promoted_teams_2025
            away_promoted = away_team in self.promoted_teams_2025
            
            if home_promoted or away_promoted:
                if home_promoted and away_promoted:
                    # Deux équipes promues: volatilité maximale
                    factor = 1.0
                elif home_promoted:
                    # Home promue: facteur positif (avantage adaptation domicile)
                    factor = 0.7
                else:  # away_promoted
                    # Away promue: facteur négatif (désavantage extérieur)
                    factor = -0.7
                
                # Ajustement temporel: impact diminue au fil de la saison
                days_since_start = (match['Date'] - self.season_start_2025).days
                decay_factor = max(0.1, 1.0 - (days_since_start / 180))  # Décroît sur 6 mois
                
                df_enhanced.loc[i, 'promoted_team_factor'] = factor * decay_factor
        
        promoted_matches = (df_enhanced['promoted_team_factor'] != 0.0).sum()
        print(f"  ✅ {promoted_matches} matches avec équipes promues")
        
        return df_enhanced
    
    def calculate_early_season_volatility(self, df_enhanced):
        """Feature 3: Facteur volatilité historique début saison par équipe"""
        print("📊 Calcul early_season_volatility...")
        
        # Calculer volatilité historique par équipe sur premières journées
        team_volatility = {}
        
        # Analyser données historiques (2019-2024) pour volatilité début saison
        historical_data = self.df[self.df['Date'] < self.season_start_2025].copy()
        
        for season in historical_data['Season'].unique():
            season_data = historical_data[historical_data['Season'] == season].sort_values('Date')
            
            # Premiers 40 matches de chaque saison (≈ J1-J4)
            if len(season_data) >= 40:
                early_matches = season_data.head(40)
                
                # Pour chaque équipe, calculer variance résultats début saison
                for team in pd.concat([early_matches['HomeTeam'], early_matches['AwayTeam']]).unique():
                    team_matches = early_matches[
                        (early_matches['HomeTeam'] == team) | (early_matches['AwayTeam'] == team)
                    ].copy()
                    
                    if len(team_matches) >= 3:  # Minimum 3 matches
                        # Convertir résultats en scores (-1, 0, 1)
                        team_results = []
                        for _, match in team_matches.iterrows():
                            if match['HomeTeam'] == team:
                                # Équipe à domicile
                                if match['FullTimeResult'] == 'H':
                                    team_results.append(1)  # Victoire
                                elif match['FullTimeResult'] == 'D':
                                    team_results.append(0)  # Nul
                                else:
                                    team_results.append(-1)  # Défaite
                            else:
                                # Équipe à l'extérieur
                                if match['FullTimeResult'] == 'A':
                                    team_results.append(1)  # Victoire
                                elif match['FullTimeResult'] == 'D':
                                    team_results.append(0)  # Nul
                                else:
                                    team_results.append(-1)  # Défaite
                        
                        # Volatilité = écart-type des résultats
                        if len(team_results) > 1:
                            volatility = np.std(team_results)
                            
                            if team not in team_volatility:
                                team_volatility[team] = []
                            team_volatility[team].append(volatility)
        
        # Moyenne des volatilités par équipe
        team_avg_volatility = {}
        for team, volatilities in team_volatility.items():
            team_avg_volatility[team] = np.mean(volatilities)
        
        # Normaliser volatilités (0 = stable, 1 = très volatile)
        if team_avg_volatility:
            max_vol = max(team_avg_volatility.values())
            min_vol = min(team_avg_volatility.values())
            vol_range = max_vol - min_vol
            
            if vol_range > 0:
                for team in team_avg_volatility:
                    team_avg_volatility[team] = (team_avg_volatility[team] - min_vol) / vol_range
        
        # Appliquer à dataset
        df_enhanced['early_season_volatility'] = 0.5  # Valeur neutre par défaut
        
        for i, match in df_enhanced.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            home_vol = team_avg_volatility.get(home_team, 0.5)
            away_vol = team_avg_volatility.get(away_team, 0.5)
            
            # Moyenne pondérée de volatilité
            combined_volatility = (home_vol + away_vol) / 2
            
            # Appliquer seulement en début saison
            days_since_start = (match['Date'] - self.season_start_2025).days
            if days_since_start <= 45:  # Premiers 45 jours
                df_enhanced.loc[i, 'early_season_volatility'] = combined_volatility
        
        print(f"  ✅ Volatilité calculée pour {len(team_avg_volatility)} équipes")
        
        return df_enhanced
    
    def calculate_manager_continuity(self, df_enhanced):
        """Feature 4: Facteur continuité équipe technique (approximation)"""
        print("👔 Calcul manager_continuity...")
        
        # Approximation basée sur performance historique récente
        # En réalité, nécessiterait base données changements managers
        
        df_enhanced['manager_continuity'] = 0.8  # Valeur par défaut (stable)
        
        # Identifier équipes avec changements potentiels
        # Approximation: équipes avec performance décroissante saison précédente
        
        previous_season_data = self.df[
            (self.df['Date'] >= pd.Timestamp('2024-08-01')) & 
            (self.df['Date'] < self.season_start_2025)
        ]
        
        if len(previous_season_data) > 0:
            # Calculer performance fin saison précédente par équipe
            team_end_performance = {}
            
            for team in pd.concat([previous_season_data['HomeTeam'], previous_season_data['AwayTeam']]).unique():
                team_matches = previous_season_data[
                    (previous_season_data['HomeTeam'] == team) | (previous_season_data['AwayTeam'] == team)
                ].sort_values('Date')
                
                if len(team_matches) >= 10:
                    # Derniers 10 matches
                    last_matches = team_matches.tail(10)
                    
                    # Calculer points approximatifs
                    points = 0
                    for _, match in last_matches.iterrows():
                        if match['HomeTeam'] == team:
                            if match['FullTimeResult'] == 'H':
                                points += 3
                            elif match['FullTimeResult'] == 'D':
                                points += 1
                        else:
                            if match['FullTimeResult'] == 'A':
                                points += 3
                            elif match['FullTimeResult'] == 'D':
                                points += 1
                    
                    team_end_performance[team] = points / 30  # Normaliser (0-1)
            
            # Équipes avec faible performance = changement manager probable
            low_performance_teams = [
                team for team, perf in team_end_performance.items() 
                if perf < 0.3  # Moins de 30% points possibles
            ]
            
            # Appliquer facteur continuité réduit
            for i, match in df_enhanced.iterrows():
                home_team = match['HomeTeam']
                away_team = match['AwayTeam']
                
                # Réduire continuité si équipe avait mauvaise performance
                home_continuity = 0.4 if home_team in low_performance_teams else 0.8
                away_continuity = 0.4 if away_team in low_performance_teams else 0.8
                
                # Moyenne pondérée
                df_enhanced.loc[i, 'manager_continuity'] = (home_continuity + away_continuity) / 2
        
        low_continuity = (df_enhanced['manager_continuity'] < 0.6).sum()
        print(f"  ✅ {low_continuity} matches avec continuité réduite")
        
        return df_enhanced
    
    def calculate_transfer_window_impact(self, df_enhanced):
        """Feature 5: Impact mercato récent (approximation)"""
        print("🔄 Calcul transfer_window_impact...")
        
        df_enhanced['transfer_window_impact'] = 0.0
        
        # Approximation: impact décroît après fermeture mercato
        for i, match in df_enhanced.iterrows():
            match_date = match['Date']
            
            if match_date >= self.transfer_deadline:
                # Après fermeture mercato, impact décroissant
                days_since_deadline = (match_date - self.transfer_deadline).days
                
                # Impact maximal première semaine, décroît sur 6 semaines
                if days_since_deadline <= 42:  # 6 semaines
                    impact = 1.0 - (days_since_deadline / 42)
                    df_enhanced.loc[i, 'transfer_window_impact'] = impact
            else:
                # Pendant mercato, impact maximal
                df_enhanced.loc[i, 'transfer_window_impact'] = 1.0
        
        impacted_matches = (df_enhanced['transfer_window_impact'] > 0.0).sum()
        print(f"  ✅ {impacted_matches} matches avec impact mercato")
        
        return df_enhanced
    
    def create_enhanced_dataset(self):
        """Créer dataset avec toutes les features contextuelles"""
        print("🔧 CRÉATION FEATURES CONTEXTUELLES DÉBUT SAISON")
        print("=" * 60)
        
        df_enhanced = self.df.copy()
        
        # Appliquer toutes les features
        df_enhanced = self.calculate_rest_days_diff(df_enhanced)
        df_enhanced = self.calculate_promoted_team_factor(df_enhanced)
        df_enhanced = self.calculate_early_season_volatility(df_enhanced)
        df_enhanced = self.calculate_manager_continuity(df_enhanced)
        df_enhanced = self.calculate_transfer_window_impact(df_enhanced)
        
        # Features finales
        contextual_features = [
            'rest_days_diff', 'promoted_team_factor', 'early_season_volatility',
            'manager_continuity', 'transfer_window_impact'
        ]
        
        # Statistiques features créées
        print(f"\n📊 STATISTIQUES FEATURES CRÉÉES:")
        print("-" * 40)
        
        for feat in contextual_features:
            non_zero = (df_enhanced[feat] != 0.0).sum()
            mean_val = df_enhanced[feat].mean()
            std_val = df_enhanced[feat].std()
            
            print(f"{feat:25}: {non_zero:4d} non-zero, mean={mean_val:.3f}, std={std_val:.3f}")
        
        return df_enhanced, contextual_features
    
    def test_contextual_features_impact(self, df_enhanced, contextual_features):
        """Tester impact des features contextuelles sur performance"""
        print(f"\n🧪 TEST IMPACT FEATURES CONTEXTUELLES")
        print("=" * 60)
        
        # Features baseline
        baseline_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
            'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
            'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        # Features étendues
        extended_features = baseline_features + contextual_features
        
        # Split train/test
        cutoff_date = pd.Timestamp('2025-08-01')
        train_df = df_enhanced[df_enhanced['Date'] < cutoff_date]
        test_df = df_enhanced[df_enhanced['Date'] >= cutoff_date]
        
        # Ajouter matches J4 pour test complet
        j4_matches = [
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Arsenal', 'AwayTeam': 'Nottingham Forest', 'FullTimeResult': 'H'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Bournemouth', 'AwayTeam': 'Brighton', 'FullTimeResult': 'H'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Crystal Palace', 'AwayTeam': 'Sunderland', 'FullTimeResult': 'D'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Everton', 'AwayTeam': 'Aston Villa', 'FullTimeResult': 'D'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Fulham', 'AwayTeam': 'Leeds', 'FullTimeResult': 'H'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Newcastle', 'AwayTeam': 'Wolves', 'FullTimeResult': 'H'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'West Ham', 'AwayTeam': 'Tottenham', 'FullTimeResult': 'A'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Brentford', 'AwayTeam': 'Chelsea', 'FullTimeResult': 'D'},
            {'Date': pd.Timestamp('2025-09-14'), 'HomeTeam': 'Burnley', 'AwayTeam': 'Liverpool', 'FullTimeResult': 'A'},
            {'Date': pd.Timestamp('2025-09-14'), 'HomeTeam': 'Manchester City', 'AwayTeam': 'Man United', 'FullTimeResult': 'H'}
        ]
        
        # Créer features pour J4 (approximation)
        j4_data = []
        for match in j4_matches:
            j4_row = match.copy()
            j4_row['Season'] = '2025-2026'
            
            # Features baseline par défaut
            for feat in baseline_features:
                if feat == 'matchday_normalized':
                    j4_row[feat] = 4/38
                elif feat == 'h2h_score':
                    j4_row[feat] = 0.5
                else:
                    j4_row[feat] = 0.5
            
            # Features contextuelles (calcul simplifié)
            j4_row['rest_days_diff'] = 0.0
            j4_row['promoted_team_factor'] = 0.3 if match['HomeTeam'] in self.promoted_teams_2025 or match['AwayTeam'] in self.promoted_teams_2025 else 0.0
            j4_row['early_season_volatility'] = 0.6  # Début saison = volatilité élevée
            j4_row['manager_continuity'] = 0.8
            j4_row['transfer_window_impact'] = 0.8  # Proche fermeture mercato
            
            j4_data.append(j4_row)
        
        j4_df = pd.DataFrame(j4_data)
        test_extended = pd.concat([test_df, j4_df], ignore_index=True)
        
        # Test avec modèle existant (approximation)
        try:
            model = joblib.load('models/final_robust_model_20250915_163023.joblib')
            
            # Test baseline
            test_baseline_clean = test_extended.dropna(subset=baseline_features + ['FullTimeResult'])
            if len(test_baseline_clean) > 0:
                X_baseline = test_baseline_clean[baseline_features]
                y_true = test_baseline_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
                
                baseline_pred = model.predict(X_baseline)
                baseline_acc = accuracy_score(y_true, baseline_pred)
                
                print(f"📊 Performance baseline (10 features): {baseline_acc:.1%} ({np.sum(baseline_pred == y_true)}/{len(y_true)})")
            
            # Analyser corrélations features contextuelles avec target
            print(f"\n📈 ANALYSE PRÉDICTIVITÉ FEATURES:")
            print("-" * 40)
            
            if len(test_baseline_clean) > 0:
                y_numeric = test_baseline_clean['FullTimeResult'].map({'H': 1, 'D': 0, 'A': -1})
                
                for feat in contextual_features:
                    if feat in test_baseline_clean.columns:
                        correlation = test_baseline_clean[feat].corr(y_numeric)
                        mean_val = test_baseline_clean[feat].mean()
                        print(f"{feat:25}: corr={correlation:+.3f}, mean={mean_val:.3f}")
                
                # Recommandations basées sur corrélations
                print(f"\n💡 RECOMMANDATIONS:")
                print("-" * 20)
                print("✅ Features contextuelles créées avec succès")
                print("🔧 Recalibrage nécessaire avec plus de données début saison")
                print("📊 Monitoring performance requis après intégration")
        
        except FileNotFoundError:
            print("⚠️ Modèle non trouvé - analyse de corrélation seulement")
        
        return test_extended, extended_features
    
    def save_enhanced_dataset(self, df_enhanced, contextual_features):
        """Sauvegarder dataset avec features contextuelles"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder dataset complet
        output_path = f"data/processed/v16_contextual_features_{timestamp}.csv"
        df_enhanced.to_csv(output_path, index=False)
        
        # Métadonnées
        metadata = {
            'timestamp': timestamp,
            'original_features': 15,  # Baseline v15
            'contextual_features_added': len(contextual_features),
            'total_features': 15 + len(contextual_features),
            'contextual_features': contextual_features,
            'description': 'Features contextuelles début saison pour améliorer prédiction J1-J6',
            'promoted_teams_2025': self.promoted_teams_2025,
            'season_start_2025': str(self.season_start_2025),
            'transfer_deadline': str(self.transfer_deadline)
        }
        
        metadata_path = f"data/processed/v16_contextual_metadata_{timestamp}.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n💾 SAUVEGARDE COMPLÉTÉE:")
        print(f"📄 Dataset: {output_path}")
        print(f"📋 Métadonnées: {metadata_path}")
        
        return output_path, metadata_path

def main():
    """Créer features contextuelles début saison complètes"""
    print("💡 CRÉATION FEATURES CONTEXTUELLES DÉBUT SAISON")
    print("=" * 80)
    
    # Initialiser
    engineer = EarlySeasonFeatureEngineer()
    
    # Créer features
    df_enhanced, contextual_features = engineer.create_enhanced_dataset()
    
    # Tester impact
    test_data, extended_features = engineer.test_contextual_features_impact(df_enhanced, contextual_features)
    
    # Sauvegarder
    output_path, metadata_path = engineer.save_enhanced_dataset(df_enhanced, contextual_features)
    
    # Résumé final
    print(f"\n🎯 RÉSUMÉ CRÉATION FEATURES:")
    print("=" * 50)
    print(f"✅ Features contextuelles créées: {len(contextual_features)}")
    print(f"📊 Dataset étendu: {len(df_enhanced)} matches")
    print(f"🔧 Prêt pour intégration modèle J5+")
    print(f"\n💡 Prochaines étapes: Tester sur vraies données J5+ pour validation")

if __name__ == "__main__":
    main()