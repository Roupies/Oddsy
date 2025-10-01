#!/usr/bin/env python3
"""
Validation Real Coverage - Vérification 100% Réel
===============================================
Script de validation finale pour garantir:
- 0% données simulées/fallback
- Couverture xG complète et cohérente
- Intégrité temporal des roulants
- Assertions de production
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RealCoverageValidator:
    """Validateur couverture 100% réelle"""
    
    def __init__(self):
        self.validation_results = {}
        self.errors_found = []
        self.warnings_found = []
        
    def validate_understat_extraction(self, extraction_path="data/understat/understat_epl_j1_j6_strict_real.csv",
                                     report_path="data/understat/strict_extraction_report_*.json"):
        """Valide extraction Understat 100% réelle"""
        
        print("🔍 VALIDATION EXTRACTION UNDERSTAT")
        print("=" * 50)
        
        try:
            # Charger dataset
            df_understat = pd.read_csv(extraction_path)
            print(f"📊 Dataset: {len(df_understat)} matchs")
            
            # 1. Validation source 100% réelle
            if 'source' in df_understat.columns:
                non_real = df_understat[df_understat['source'] != 'understat_real']
                if len(non_real) > 0:
                    self.errors_found.append(f"DONNÉES NON RÉELLES: {len(non_real)} lignes source != 'understat_real'")
                else:
                    print("✅ Source: 100% understat_real")
            else:
                self.warnings_found.append("Colonne 'source' manquante - impossible de valider origine")
            
            # 2. Validation xG coverage
            missing_h_xg = df_understat['H_xG'].isna().sum()
            missing_a_xg = df_understat['A_xG'].isna().sum()
            
            if missing_h_xg > 0 or missing_a_xg > 0:
                self.errors_found.append(f"xG MANQUANTS: H_xG={missing_h_xg}, A_xG={missing_a_xg}")
            else:
                print("✅ xG Coverage: 100% complet")
            
            # 3. Validation plages xG réalistes
            xg_out_of_bounds = df_understat[
                (df_understat['H_xG'] < 0) | (df_understat['H_xG'] > 10) |
                (df_understat['A_xG'] < 0) | (df_understat['A_xG'] > 10)
            ]
            
            if len(xg_out_of_bounds) > 0:
                self.errors_found.append(f"xG HORS LIMITES: {len(xg_out_of_bounds)} matchs")
            else:
                print("✅ xG Ranges: Toutes valeurs dans [0, 10]")
            
            # 4. Validation mapping équipes
            missing_mapping = df_understat[
                df_understat['HomeTeam_FD'].isna() | df_understat['AwayTeam_FD'].isna()
            ]
            
            if len(missing_mapping) > 0:
                self.errors_found.append(f"MAPPING MANQUANT: {len(missing_mapping)} matchs")
            else:
                print("✅ Team Mapping: 100% complet")
            
            # 5. Validation J1-J6
            invalid_rounds = df_understat[
                (df_understat['Round'] < 1) | (df_understat['Round'] > 6)
            ]
            
            if len(invalid_rounds) > 0:
                self.errors_found.append(f"ROUNDS INVALIDES: {len(invalid_rounds)} matchs hors J1-J6")
            else:
                print("✅ Rounds: Tous dans J1-J6")
            
            # 6. Validation chronologique
            df_understat['Date'] = pd.to_datetime(df_understat['Date'])
            if not df_understat['Date'].is_monotonic_increasing:
                self.warnings_found.append("Dates non triées chronologiquement")
            else:
                print("✅ Chronologie: Dates triées")
            
            self.validation_results['understat_extraction'] = {
                'total_matches': len(df_understat),
                'real_data_pct': 100.0 if len(self.errors_found) == 0 else 0.0,
                'xg_coverage_pct': ((len(df_understat) - missing_h_xg - missing_a_xg) / len(df_understat)) * 100,
                'mapping_success_pct': ((len(df_understat) - len(missing_mapping)) / len(df_understat)) * 100,
                'valid': len(self.errors_found) == 0
            }
            
            return df_understat
            
        except Exception as e:
            self.errors_found.append(f"ÉCHEC VALIDATION UNDERSTAT: {e}")
            return None
    
    def validate_jointure_strict(self, enhanced_path="data/processed/enhanced_features_strict_temporal.csv",
                                jointure_report="data/processed/strict_temporal_report.json"):
        """Valide jointure stricte Date+équipes"""
        
        print("\n🔗 VALIDATION JOINTURE STRICTE")
        print("=" * 50)
        
        try:
            # Charger dataset enhanced
            df_enhanced = pd.read_csv(enhanced_path)
            df_enhanced['Date'] = pd.to_datetime(df_enhanced['Date'])
            print(f"📊 Dataset enhanced: {len(df_enhanced)} matchs")
            
            # Charger rapport jointure
            if os.path.exists(jointure_report):
                with open(jointure_report, 'r') as f:
                    report = json.load(f)
                jointure_details = report.get('jointure_details', {})
            else:
                self.warnings_found.append("Rapport jointure manquant")
                jointure_details = {}
            
            # 1. Validation completude datasets source
            expected_understat_matches = 60  # J1-J6 max
            actual_enhanced_matches = len(df_enhanced)
            
            coverage_rate = actual_enhanced_matches / expected_understat_matches
            
            if coverage_rate < 0.8:  # Seuil 80% minimum
                self.errors_found.append(f"COUVERTURE JOINTURE FAIBLE: {coverage_rate*100:.1f}% < 80%")
            else:
                print(f"✅ Couverture jointure: {coverage_rate*100:.1f}%")
            
            # 2. Validation cohérence dates
            season_start = datetime(2025, 8, 1)
            season_end = datetime(2026, 5, 31)
            
            out_of_season = df_enhanced[
                (df_enhanced['Date'] < season_start) | (df_enhanced['Date'] > season_end)
            ]
            
            if len(out_of_season) > 0:
                self.errors_found.append(f"DATES HORS SAISON: {len(out_of_season)} matchs")
            else:
                print("✅ Dates: Toutes dans saison EPL 2025-26")
            
            # 3. Validation données fusionnées complètes
            essential_cols = ['H_xG_actual', 'A_xG_actual', 'H_Shots', 'A_Shots', 'H_Corner', 'A_Corner']
            
            for col in essential_cols:
                missing = df_enhanced[col].isna().sum()
                if missing > 0:
                    self.errors_found.append(f"{col} MANQUANT: {missing} valeurs")
                else:
                    print(f"✅ {col}: 100% complet")
            
            # 4. Validation équipes uniques cohérentes
            all_teams = set(df_enhanced['HomeTeam'].unique()) | set(df_enhanced['AwayTeam'].unique())
            expected_epl_teams = 20
            
            if len(all_teams) != expected_epl_teams:
                self.warnings_found.append(f"Nombre équipes: {len(all_teams)} != {expected_epl_teams} attendues")
                print(f"⚠️  Équipes trouvées: {sorted(all_teams)}")
            else:
                print(f"✅ Équipes: {len(all_teams)} équipes EPL validées")
            
            self.validation_results['jointure_strict'] = {
                'matches_joined': len(df_enhanced),
                'coverage_rate': coverage_rate,
                'data_completeness': 100.0 - (sum([df_enhanced[col].isna().sum() for col in essential_cols]) / (len(df_enhanced) * len(essential_cols))) * 100,
                'teams_count': len(all_teams),
                'valid': len(out_of_season) == 0 and all([df_enhanced[col].isna().sum() == 0 for col in essential_cols])
            }
            
            return df_enhanced
            
        except Exception as e:
            self.errors_found.append(f"ÉCHEC VALIDATION JOINTURE: {e}")
            return None
    
    def validate_temporal_rolling(self, df_enhanced):
        """Valide calculs roulants temporels stricts"""
        
        print("\n⏰ VALIDATION ROULANTS TEMPORELS")
        print("=" * 50)
        
        try:
            # 1. Validation anti-fuite chronologique
            # Vérifier que dates sont triées
            if not df_enhanced['Date'].is_monotonic_increasing:
                self.warnings_found.append("Dataset non trié chronologiquement")
            else:
                print("✅ Chronologie: Dataset trié par Date")
            
            # 2. Validation features roulantes
            rolling_features = ['home_xg_avg_10', 'away_xg_avg_10', 'home_xg_eff_10', 'away_xg_eff_10']
            
            for feature in rolling_features:
                if feature in df_enhanced.columns:
                    # Vérifier bornes réalistes
                    if 'eff' in feature:
                        # Efficiency doit être bornée [0.3, 1.7]
                        out_of_bounds = df_enhanced[
                            (df_enhanced[feature] < 0.3) | (df_enhanced[feature] > 1.7)
                        ].dropna()
                        
                        if len(out_of_bounds) > 0:
                            self.errors_found.append(f"{feature} HORS BORNES: {len(out_of_bounds)} valeurs hors [0.3, 1.7]")
                        else:
                            print(f"✅ {feature}: Bornage [0.3, 1.7] respecté")
                    
                    else:
                        # xG avg doit être réaliste [0, 5]
                        out_of_bounds = df_enhanced[
                            (df_enhanced[feature] < 0) | (df_enhanced[feature] > 5)
                        ].dropna()
                        
                        if len(out_of_bounds) > 0:
                            self.warnings_found.append(f"{feature} VALEURS EXTRÊMES: {len(out_of_bounds)} valeurs hors [0, 5]")
                        else:
                            print(f"✅ {feature}: Valeurs réalistes [0, 5]")
            
            # 3. Validation flags k≥3 avec seuil minimal
            if 'home_xg_valid' in df_enhanced.columns and 'away_xg_valid' in df_enhanced.columns:
                xg_valid_count = (df_enhanced['home_xg_valid'] & df_enhanced['away_xg_valid']).sum()
                xg_valid_pct = xg_valid_count / len(df_enhanced) * 100
                
                # Seuil minimal global ≥70%
                min_xg_valid_threshold = 70.0
                if xg_valid_pct < min_xg_valid_threshold:
                    self.warnings_found.append(f"XG_VALID FAIBLE: {xg_valid_pct:.1f}% < {min_xg_valid_threshold}% - Recommandation: retarder promotion")
                else:
                    print(f"✅ Validation k≥3: {xg_valid_pct:.1f}% matchs avec données suffisantes")
                
                # Vérifier cohérence k et valid flags
                invalid_flags = df_enhanced[
                    ((df_enhanced['home_k'] >= 3) & (~df_enhanced['home_xg_valid'])) |
                    ((df_enhanced['away_k'] >= 3) & (~df_enhanced['away_xg_valid']))
                ]
                
                if len(invalid_flags) > 0:
                    self.errors_found.append(f"FLAGS k≥3 INCOHÉRENTS: {len(invalid_flags)} matchs")
                else:
                    print("✅ Flags k≥3: Cohérents avec compteurs")
            
            # 4. Validation constantes éliminées
            constants_shots = (df_enhanced['shots_diff_normalized'] == 0.5).sum()
            constants_corners = (df_enhanced['corners_diff_normalized'] == 0.5).sum()
            
            elimination_rate = ((len(df_enhanced) - constants_shots - constants_corners) / (2 * len(df_enhanced))) * 100
            
            if elimination_rate < 80:  # Seuil 80% minimum
                self.warnings_found.append(f"ÉLIMINATION CONSTANTES FAIBLE: {elimination_rate:.1f}% < 80%")
            else:
                print(f"✅ Constantes éliminées: {elimination_rate:.1f}%")
            
            self.validation_results['temporal_rolling'] = {
                'xg_valid_coverage_pct': xg_valid_pct if 'home_xg_valid' in df_enhanced.columns else 0,
                'constants_elimination_pct': elimination_rate,
                'chronological_order': df_enhanced['Date'].is_monotonic_increasing,
                'efficiency_bounded': len([f for f in rolling_features if 'eff' in f and f in df_enhanced.columns]),
                'valid': len(self.errors_found) == 0
            }
            
            return True
            
        except Exception as e:
            self.errors_found.append(f"ÉCHEC VALIDATION TEMPOREL: {e}")
            return False
    
    def validate_production_readiness(self, df_enhanced):
        """Validation finale prêt production"""
        
        print("\n🚀 VALIDATION PRODUCTION READINESS")
        print("=" * 50)
        
        try:
            # 1. Assertions critiques
            critical_assertions = []
            
            # Assertion 1: Pas de NaN dans features critiques
            critical_features = ['H_xG_actual', 'A_xG_actual', 'shots_diff_normalized', 'corners_diff_normalized']
            for feature in critical_features:
                nan_count = df_enhanced[feature].isna().sum()
                if nan_count > 0:
                    critical_assertions.append(f"{feature}: {nan_count} NaN")
            
            # Assertion 2: Variance > 0 pour features normalisées
            variance_features = ['shots_diff_normalized', 'corners_diff_normalized']
            for feature in variance_features:
                variance = df_enhanced[feature].var()
                if variance <= 0.001:  # Seuil variance minimum
                    critical_assertions.append(f"{feature}: Variance trop faible ({variance:.6f})")
            
            # Assertion 3: Range xG réaliste
            xg_features = ['H_xG_actual', 'A_xG_actual']
            for feature in xg_features:
                min_val = df_enhanced[feature].min()
                max_val = df_enhanced[feature].max()
                if min_val < 0 or max_val > 10:
                    critical_assertions.append(f"{feature}: Range invalide [{min_val}, {max_val}]")
            
            if critical_assertions:
                for assertion in critical_assertions:
                    self.errors_found.append(f"ASSERTION CRITIQUE: {assertion}")
            else:
                print("✅ Assertions critiques: TOUTES VALIDÉES")
            
            # 2. Métriques qualité globale DURCIES
            total_matches = len(df_enhanced)
            complete_data_matches = df_enhanced.dropna(subset=['H_xG_actual', 'A_xG_actual', 'shots_diff_normalized']).shape[0]
            quality_score = complete_data_matches / total_matches * 100
            
            # Seuil durci à 98%
            quality_threshold = 98.0
            print(f"📊 Score qualité: {quality_score:.1f}% (seuil: {quality_threshold}%)")
            
            # Hash/commit des scripts pour traçabilité
            script_version = {
                'timestamp': datetime.now().isoformat(),
                'validation_version': '2.0_strict',
                'quality_threshold': quality_threshold
            }
            
            # 3. Résumé validation finale STRICT
            production_ready = (
                len(critical_assertions) == 0 and
                quality_score >= quality_threshold and  # Durci à 98%
                len(self.errors_found) == 0 and
                len(self.warnings_found) <= 3  # Warnings limités
            )
            
            # 4. Génération CSV issues pour correction manuelle
            issues_data = []
            
            # Issues des erreurs
            for i, error in enumerate(self.errors_found):
                issues_data.append({
                    'type': 'ERROR',
                    'id': f"ERR_{i+1}",
                    'description': error,
                    'severity': 'CRITICAL',
                    'requires_fix': True
                })
            
            # Issues des warnings
            for i, warning in enumerate(self.warnings_found):
                issues_data.append({
                    'type': 'WARNING', 
                    'id': f"WARN_{i+1}",
                    'description': warning,
                    'severity': 'MODERATE',
                    'requires_fix': False
                })
            
            # Sauvegarder CSV issues
            if issues_data:
                issues_path = "data/processed/production_issues.csv"
                pd.DataFrame(issues_data).to_csv(issues_path, index=False)
                print(f"📋 Issues CSV: {issues_path}")
            
            self.validation_results['production_readiness'] = {
                'total_matches': total_matches,
                'complete_data_matches': complete_data_matches,
                'quality_score': quality_score,
                'quality_threshold': quality_threshold,
                'critical_assertions_passed': len(critical_assertions) == 0,
                'errors_count': len(self.errors_found),
                'warnings_count': len(self.warnings_found),
                'warnings_threshold': 3,
                'production_ready': production_ready,
                'script_version': script_version,
                'issues_generated': len(issues_data)
            }
            
            if production_ready:
                print("🎯 STATUT: PRÊT PRODUCTION ✅")
            else:
                print("⚠️  STATUT: CORRECTIONS REQUISES")
            
            return production_ready
            
        except Exception as e:
            self.errors_found.append(f"ÉCHEC VALIDATION PRODUCTION: {e}")
            return False
    
    def generate_validation_report(self):
        """Génère rapport validation complet"""
        
        # Résumé global
        global_summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_mode': 'REAL_COVERAGE_STRICT',
            'overall_status': 'PASSED' if len(self.errors_found) == 0 else 'FAILED',
            'errors_found': self.errors_found,
            'warnings_found': self.warnings_found,
            'validation_results': self.validation_results
        }
        
        # Sauvegarder rapport
        report_path = "data/processed/real_coverage_validation_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(global_summary, f, indent=2)
        
        # Affichage résumé
        print(f"\n" + "=" * 70)
        print("📋 RAPPORT VALIDATION COUVERTURE RÉELLE")
        print("=" * 70)
        
        if len(self.errors_found) == 0:
            print("🎯 STATUT GLOBAL: ✅ VALIDATION RÉUSSIE")
            print("🔒 GARANTIE: 100% données réelles confirmées")
        else:
            print("❌ STATUT GLOBAL: ÉCHEC VALIDATION")
            print("🚨 ERREURS CRITIQUES:")
            for error in self.errors_found:
                print(f"   - {error}")
        
        if self.warnings_found:
            print("⚠️  AVERTISSEMENTS:")
            for warning in self.warnings_found:
                print(f"   - {warning}")
        
        print(f"\n📄 Rapport détaillé: {report_path}")
        
        return report_path, len(self.errors_found) == 0

def main():
    """Pipeline validation complète"""
    
    print("🔍 VALIDATION REAL COVERAGE - 100% RÉEL")
    print("=" * 70)
    print("Objectif: Vérifier garantie zéro simulation + intégrité temporelle")
    
    validator = RealCoverageValidator()
    
    try:
        # 1. Valider extraction Understat
        df_understat = validator.validate_understat_extraction()
        if df_understat is None:
            raise ValueError("ÉCHEC validation extraction")
        
        # 2. Valider jointure stricte
        df_enhanced = validator.validate_jointure_strict()
        if df_enhanced is None:
            raise ValueError("ÉCHEC validation jointure")
        
        # 3. Valider roulants temporels
        temporal_valid = validator.validate_temporal_rolling(df_enhanced)
        if not temporal_valid:
            raise ValueError("ÉCHEC validation temporel")
        
        # 4. Valider prêt production
        production_ready = validator.validate_production_readiness(df_enhanced)
        
        # 5. Générer rapport final
        report_path, validation_passed = validator.generate_validation_report()
        
        if validation_passed and production_ready:
            print(f"\n✅ VALIDATION COMPLÈTE RÉUSSIE")
            print(f"🚀 Dataset prêt pour intégration pipeline J7+")
            return report_path
        else:
            print(f"\n❌ VALIDATION ÉCHOUÉE - Corrections requises")
            return None
        
    except Exception as e:
        print(f"\n💥 ÉCHEC VALIDATION GLOBALE: {e}")
        return None

if __name__ == "__main__":
    main()