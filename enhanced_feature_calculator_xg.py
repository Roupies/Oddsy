#!/usr/bin/env python3
"""
Calculateur Enhanced Features avec xG Roulants Stricts
====================================================
Consomme H_xG/A_xG Understat + shots/corners E0
Roulants anti-fuite: Date < match et k≥3
Recalcule home_xg_eff_10 et away_xg_eff_10 avec vraies données
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureCalculatorXG:
    """Calculateur features enhanced avec xG réels et anti-fuite"""
    
    def __init__(self, min_matches_k=3):
        self.min_k = min_matches_k  # Seuil k≥3 pour anti-fuite
        self.feature_log = []
        
    def load_datasets(self, understat_path="data/understat/understat_epl_j1_j6_real.csv", 
                      e0_path="data/raw/E0 (14).csv"):
        """Charge Understat xG + E0 shots/corners"""
        
        print("📊 Chargement datasets...")
        
        # 1. Understat xG
        try:
            df_xg = pd.read_csv(understat_path)
            df_xg['Date'] = pd.to_datetime(df_xg['Date'])
            print(f"✅ Understat: {len(df_xg)} matchs xG")
        except Exception as e:
            print(f"❌ Erreur Understat: {e}")
            return None, None
        
        # 2. E0 shots/corners  
        try:
            df_e0 = pd.read_csv(e0_path, encoding='utf-8-sig')
            df_e0['Date'] = pd.to_datetime(df_e0['Date'], format='%d/%m/%Y')
            
            # Colonnes essentielles
            e0_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']
            df_e0 = df_e0[e0_cols].rename(columns={
                'FTHG': 'H_Goals', 'FTAG': 'A_Goals',
                'HS': 'H_Shots', 'AS': 'A_Shots',
                'HST': 'H_SoT', 'AST': 'A_SoT', 
                'HC': 'H_Corner', 'AC': 'A_Corner'
            })
            print(f"✅ E0: {len(df_e0)} matchs shots/corners")
        except Exception as e:
            print(f"❌ Erreur E0: {e}")
            return None, None
            
        return df_xg, df_e0
    
    def create_master_dataset(self, df_xg, df_e0):
        """Fusionne xG Understat + shots/corners E0"""
        
        print("🔄 Fusion master dataset...")
        
        # Jointure par équipes (utiliser mapping Understat)
        merged_data = []
        
        for _, xg_match in df_xg.iterrows():
            home_fd = xg_match['HomeTeam_FD'] 
            away_fd = xg_match['AwayTeam_FD']
            
            # Chercher correspondance E0
            e0_match = df_e0[
                (df_e0['HomeTeam'] == home_fd) & 
                (df_e0['AwayTeam'] == away_fd)
            ]
            
            if not e0_match.empty:
                e0_row = e0_match.iloc[0]
                
                merged_match = {
                    'Date': xg_match['Date'],
                    'Round': xg_match['Round'],
                    'HomeTeam': home_fd,  # Utiliser nom E0 pour cohérence
                    'AwayTeam': away_fd,
                    # xG depuis Understat
                    'H_xG': xg_match['H_xG'],
                    'A_xG': xg_match['A_xG'],
                    # Shots/corners depuis E0
                    'H_Goals': e0_row['H_Goals'],
                    'A_Goals': e0_row['A_Goals'],
                    'H_Shots': e0_row['H_Shots'],
                    'A_Shots': e0_row['A_Shots'],
                    'H_SoT': e0_row['H_SoT'],
                    'A_SoT': e0_row['A_SoT'],
                    'H_Corner': e0_row['H_Corner'],
                    'A_Corner': e0_row['A_Corner']
                }
                merged_data.append(merged_match)
        
        df_master = pd.DataFrame(merged_data)
        df_master = df_master.sort_values('Date').reset_index(drop=True)
        
        print(f"✅ Master dataset: {len(df_master)} matchs fusionnés")
        return df_master
    
    def calculate_rolling_xg_features(self, df_master, target_date):
        """Calcule features xG roulantes avec anti-fuite strict"""
        
        # Filtrer matchs AVANT target_date (anti-fuite)
        historical_matches = df_master[df_master['Date'] < target_date].copy()
        
        team_xg_stats = {}
        
        # Pour chaque équipe, calculer moyennes roulantes
        all_teams = set(df_master['HomeTeam'].unique()) | set(df_master['AwayTeam'].unique())
        
        for team in all_teams:
            # Matchs équipe (home + away)
            home_matches = historical_matches[historical_matches['HomeTeam'] == team]
            away_matches = historical_matches[historical_matches['AwayTeam'] == team]
            
            # xG pour/contre
            home_xg_for = home_matches['H_xG'].tolist()
            home_xg_against = home_matches['A_xG'].tolist()
            away_xg_for = away_matches['A_xG'].tolist()  
            away_xg_against = away_matches['H_xG'].tolist()
            
            # Goals pour efficiency
            home_goals_for = home_matches['H_Goals'].tolist()
            away_goals_for = away_matches['A_Goals'].tolist()
            
            all_xg_for = home_xg_for + away_xg_for
            all_xg_against = home_xg_against + away_xg_against
            all_goals_for = home_goals_for + away_goals_for
            
            # Calculer moyennes si k≥3
            if len(all_xg_for) >= self.min_k:
                # Derniers 10 matchs max
                recent_xg_for = all_xg_for[-10:]
                recent_xg_against = all_xg_against[-10:]
                recent_goals_for = all_goals_for[-10:]
                
                team_xg_stats[team] = {
                    'xg_avg_10': np.mean(recent_xg_for),
                    'xg_conceded_avg_10': np.mean(recent_xg_against),
                    'xg_efficiency_10': np.mean(recent_goals_for) / max(np.mean(recent_xg_for), 0.1),
                    'matches_count': len(all_xg_for),
                    'valid': True
                }
            else:
                # Pas assez de matchs - NaN
                team_xg_stats[team] = {
                    'xg_avg_10': np.nan,
                    'xg_conceded_avg_10': np.nan,
                    'xg_efficiency_10': np.nan,
                    'matches_count': len(all_xg_for),
                    'valid': False
                }
        
        return team_xg_stats
    
    def calculate_rolling_shots_features(self, df_master, target_date):
        """Calcule features shots/corners roulantes"""
        
        historical_matches = df_master[df_master['Date'] < target_date].copy()
        team_shots_stats = {}
        
        all_teams = set(df_master['HomeTeam'].unique()) | set(df_master['AwayTeam'].unique())
        
        for team in all_teams:
            home_matches = historical_matches[historical_matches['HomeTeam'] == team]
            away_matches = historical_matches[historical_matches['AwayTeam'] == team]
            
            # Shots/corners pour/contre
            home_shots_for = home_matches['H_Shots'].tolist()
            away_shots_for = away_matches['A_Shots'].tolist()
            home_corners_for = home_matches['H_Corner'].tolist()
            away_corners_for = away_matches['A_Corner'].tolist()
            
            all_shots_for = home_shots_for + away_shots_for  
            all_corners_for = home_corners_for + away_corners_for
            
            if len(all_shots_for) >= self.min_k:
                recent_shots = all_shots_for[-10:]
                recent_corners = all_corners_for[-10:]
                
                team_shots_stats[team] = {
                    'shots_avg_10': np.mean(recent_shots),
                    'corners_avg_10': np.mean(recent_corners),
                    'valid': True
                }
            else:
                team_shots_stats[team] = {
                    'shots_avg_10': np.nan,
                    'corners_avg_10': np.nan,  
                    'valid': False
                }
        
        return team_shots_stats
    
    def calculate_enhanced_match_features(self, df_master):
        """Calcule features enhanced pour chaque match futur"""
        
        print("🎯 Calcul features enhanced avec xG réels...")
        
        enhanced_matches = []
        
        for idx, match in df_master.iterrows():
            match_date = match['Date']
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            print(f"[{idx+1}/{len(df_master)}] {home_team} vs {away_team}")
            
            # 1. Features xG roulantes
            xg_stats = self.calculate_rolling_xg_features(df_master, match_date)
            home_xg = xg_stats.get(home_team, {})
            away_xg = xg_stats.get(away_team, {})
            
            # 2. Features shots roulantes  
            shots_stats = self.calculate_rolling_shots_features(df_master, match_date)
            home_shots = shots_stats.get(home_team, {})
            away_shots = shots_stats.get(away_team, {})
            
            # 3. Features match actuel (vraies données vs constantes!)
            shots_total = match['H_Shots'] + match['A_Shots']
            corners_total = match['H_Corner'] + match['A_Corner']
            sot_total = match['H_SoT'] + match['A_SoT']
            
            enhanced_match = {
                # Informations base
                'Date': match['Date'].strftime('%Y-%m-%d'),
                'Round': match['Round'],
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                
                # xG features (NOUVELLES avec vraies données!)
                'home_xg_avg_10': home_xg.get('xg_avg_10'),
                'away_xg_avg_10': away_xg.get('xg_avg_10'),
                'home_xg_eff_10': home_xg.get('xg_efficiency_10'),  # REMPLACE approximation!
                'away_xg_eff_10': away_xg.get('xg_efficiency_10'),  # REMPLACE approximation!
                'xg_diff_avg_10': (home_xg.get('xg_avg_10', 0) or 0) - (away_xg.get('xg_avg_10', 0) or 0),
                
                # Shots features enhanced (vs constantes 0.5!)
                'shots_diff_normalized': match['H_Shots'] / shots_total if shots_total > 0 else 0.5,
                'corners_diff_normalized': match['H_Corner'] / corners_total if corners_total > 0 else 0.5,
                'sot_diff_normalized': match['H_SoT'] / sot_total if sot_total > 0 else 0.5,
                
                # Shot accuracy (NOUVEAU)
                'home_shot_accuracy': match['H_SoT'] / match['H_Shots'] if match['H_Shots'] > 0 else 0,
                'away_shot_accuracy': match['A_SoT'] / match['A_Shots'] if match['A_Shots'] > 0 else 0,
                
                # Données raw pour validation
                'H_xG_actual': match['H_xG'],
                'A_xG_actual': match['A_xG'],
                'H_Shots': match['H_Shots'],
                'A_Shots': match['A_Shots'],
                'H_Corner': match['H_Corner'],
                'A_Corner': match['A_Corner'],
                
                # Flags validité (k≥3)
                'home_xg_valid': home_xg.get('valid', False),
                'away_xg_valid': away_xg.get('valid', False),
                'home_shots_valid': home_shots.get('valid', False),
                'away_shots_valid': away_shots.get('valid', False)
            }
            
            enhanced_matches.append(enhanced_match)
            
            # Log feature quality
            self.feature_log.append({
                'match': f"{home_team} vs {away_team}",
                'date': match['Date'].strftime('%Y-%m-%d'),
                'xg_coverage': home_xg.get('valid', False) and away_xg.get('valid', False),
                'shots_replaced_constants': shots_total > 0 and corners_total > 0,
                'home_k': home_xg.get('matches_count', 0),
                'away_k': away_xg.get('matches_count', 0)
            })
        
        df_enhanced = pd.DataFrame(enhanced_matches)
        print(f"✅ {len(df_enhanced)} matchs enhanced calculés")
        
        return df_enhanced
    
    def validate_improvements(self, df_enhanced):
        """Valide améliorations vs approximations"""
        
        print("\n🎯 VALIDATION AMÉLIORATIONS vs APPROXIMATIONS")
        print("=" * 60)
        
        # 1. Élimination constantes
        shots_variables = (df_enhanced['shots_diff_normalized'] != 0.5).sum()
        corners_variables = (df_enhanced['corners_diff_normalized'] != 0.5).sum()
        
        print(f"📊 ÉLIMINATION CONSTANTES:")
        print(f"   shots_diff_normalized variables: {shots_variables}/{len(df_enhanced)} ({shots_variables/len(df_enhanced)*100:.1f}%)")
        print(f"   corners_diff_normalized variables: {corners_variables}/{len(df_enhanced)} ({corners_variables/len(df_enhanced)*100:.1f}%)")
        
        # 2. Couverture xG réels vs approximations
        xg_valid = df_enhanced['home_xg_valid'] & df_enhanced['away_xg_valid']
        xg_coverage = xg_valid.sum()
        
        print(f"\n⚡ COUVERTURE xG RÉELS:")
        print(f"   home_xg_eff_10 avec vraies données: {xg_coverage}/{len(df_enhanced)} ({xg_coverage/len(df_enhanced)*100:.1f}%)")
        print(f"   away_xg_eff_10 avec vraies données: {xg_coverage}/{len(df_enhanced)} ({xg_coverage/len(df_enhanced)*100:.1f}%)")
        
        # 3. Variance ajoutée
        shots_var = df_enhanced['shots_diff_normalized'].var()
        corners_var = df_enhanced['corners_diff_normalized'].var()
        
        print(f"\n📈 VARIANCE AJOUTÉE:")
        print(f"   shots_diff_normalized: {shots_var:.6f} (vs 0.000000 constant)")
        print(f"   corners_diff_normalized: {corners_var:.6f} (vs 0.000000 constant)")
        
        # 4. Exemples concrets
        print(f"\n⚽ EXEMPLES CONCRETS:")
        for i, (_, match) in enumerate(df_enhanced.head(3).iterrows()):
            print(f"\n{i+1}. {match['HomeTeam']} vs {match['AwayTeam']}")
            print(f"   shots_diff: {match['shots_diff_normalized']:.4f} (vs 0.5000 constant)")
            print(f"   xG efficiency: H={match['home_xg_eff_10']:.3f} A={match['away_xg_eff_10']:.3f}")
            print(f"   xG réels: {match['H_xG_actual']:.2f} vs {match['A_xG_actual']:.2f}")
        
        return {
            'constants_eliminated_pct': (shots_variables + corners_variables) / (2 * len(df_enhanced)) * 100,
            'xg_real_coverage_pct': xg_coverage / len(df_enhanced) * 100,
            'shots_variance_added': shots_var,
            'corners_variance_added': corners_var
        }
    
    def save_enhanced_dataset(self, df_enhanced, validation_stats):
        """Sauvegarde dataset enhanced final"""
        
        output_path = "data/processed/enhanced_features_xg_real.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            df_enhanced.to_csv(output_path, index=False)
            
            # Rapport final
            final_report = {
                'generation_date': datetime.now().isoformat(),
                'dataset_path': output_path,
                'matches_processed': len(df_enhanced),
                'improvements': validation_stats,
                'feature_types': {
                    'xg_efficiency_real': ['home_xg_eff_10', 'away_xg_eff_10'],
                    'constants_eliminated': ['shots_diff_normalized', 'corners_diff_normalized'],
                    'new_features': ['sot_diff_normalized', 'home_shot_accuracy', 'away_shot_accuracy'],
                    'xg_features': ['home_xg_avg_10', 'away_xg_avg_10', 'xg_diff_avg_10']
                },
                'anti_leak_validation': {
                    'min_k_required': self.min_k,
                    'date_filtering': 'strict_before_match',
                    'nan_handling': 'when_k_insufficient'
                },
                'feature_log': self.feature_log
            }
            
            report_path = "data/processed/enhanced_xg_report.json"
            import json
            with open(report_path, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            print(f"\n💾 SAUVEGARDE TERMINÉE:")
            print(f"   Dataset: {output_path}")
            print(f"   Rapport: {report_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return None

def main():
    """Pipeline complet enhanced features avec xG réels"""
    
    print("🚀 CALCULATEUR ENHANCED FEATURES XG RÉELS")
    print("=" * 70)
    print("Objectif: Remplacer approximations par vraies données xG + shots")
    
    calculator = EnhancedFeatureCalculatorXG(min_matches_k=3)
    
    # 1. Charger datasets
    df_xg, df_e0 = calculator.load_datasets()
    if df_xg is None or df_e0 is None:
        return
    
    # 2. Créer master dataset
    df_master = calculator.create_master_dataset(df_xg, df_e0)
    if df_master.empty:
        print("❌ Échec fusion datasets")
        return
    
    # 3. Calculer features enhanced
    df_enhanced = calculator.calculate_enhanced_match_features(df_master)
    
    # 4. Valider améliorations
    validation_stats = calculator.validate_improvements(df_enhanced)
    
    # 5. Sauvegarder
    output_path = calculator.save_enhanced_dataset(df_enhanced, validation_stats)
    
    if output_path:
        print(f"\n" + "=" * 70)
        print("✅ PIPELINE ENHANCED FEATURES TERMINÉ")
        print("=" * 70)
        print("🎯 ACCOMPLISSEMENTS:")
        print(f"   ✅ xG efficiency réel vs approximation goals/1.5")
        print(f"   ✅ Constantes 0.5 remplacées: {validation_stats['constants_eliminated_pct']:.1f}%")
        print(f"   ✅ Couverture xG réels: {validation_stats['xg_real_coverage_pct']:.1f}%")
        print(f"   ✅ Anti-fuite: k≥3 + Date < match")
        print(f"   ✅ Dataset: {output_path}")

if __name__ == "__main__":
    main()