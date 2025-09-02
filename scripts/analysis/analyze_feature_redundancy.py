#!/usr/bin/env python3
"""
analyze_feature_redundancy.py

ANALYSE COMPLÈTE DE REDONDANCE DES 27 FEATURES v2.2
Basé sur l'analyse critique de Claude Opus

OBJECTIF:
- Identifier les features redondantes parmi les 27 actuelles
- Passer de 27 → 10-12 features optimales
- Préparer base saine pour features contextuelles

USAGE:
python scripts/analysis/analyze_feature_redundancy.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import (
    VarianceThreshold, 
    mutual_info_classif,
    SelectKBest,
    RFE,
    RFECV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

class FeatureRedundancyAnalyzer:
    """
    Analyseur de redondance des features pour Oddsy v2.2
    """
    
    def __init__(self, correlation_threshold=0.80, variance_threshold=0.01):
        self.corr_threshold = correlation_threshold
        self.var_threshold = variance_threshold
        self.results = {}
        self.logger = setup_logging()
        
    def load_data(self, filepath='data/processed/v13_xg_safe_features.csv'):
        """
        Charger les données v2.2 avec 27 features
        """
        self.logger.info("📊 CHARGEMENT DES DONNÉES v2.2")
        self.logger.info("="*60)
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        self.logger.info(f"✅ Données chargées: {len(df)} matches")
        
        # Liste exacte des 27 features v2.2 (basée sur v13_xg_safe_features.csv)
        self.feature_names = [
            col for col in df.columns 
            if col not in ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult']
        ]
        
        self.logger.info(f"📈 Features détectées: {len(self.feature_names)}")
        for i, feat in enumerate(self.feature_names, 1):
            self.logger.info(f"  {i:2d}. {feat}")
        
        # Préparer X et y avec notre split temporel
        train_end = pd.to_datetime('2024-05-19')  # Fin training v2.1
        test_start = pd.to_datetime('2024-08-16')  # Début test 2024-2025
        
        # Filtrer données non-manquantes
        valid_data = df.dropna(subset=self.feature_names)
        self.logger.info(f"📊 Matches avec toutes features: {len(valid_data)}")
        
        # Split temporel
        train_data = valid_data[valid_data['Date'] <= train_end].copy()
        test_data = valid_data[(valid_data['Date'] >= test_start)].copy()
        
        self.logger.info(f"📊 Train: {len(train_data)} matches (jusqu'à {train_end.strftime('%Y-%m-%d')})")
        self.logger.info(f"📊 Test: {len(test_data)} matches (depuis {test_start.strftime('%Y-%m-%d')})")
        
        # Préparer matrices
        self.X_train = train_data[self.feature_names].values
        self.X_test = test_data[self.feature_names].values
        
        # Target encoding: H->0, D->1, A->2
        self.y_train = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
        self.y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
        
        # Handle any remaining NaN with 0.5 (neutral value)
        self.X_train = np.nan_to_num(self.X_train, nan=0.5)
        self.X_test = np.nan_to_num(self.X_test, nan=0.5)
        
        self.logger.info(f"✅ Données prêtes: X_train {self.X_train.shape}, X_test {self.X_test.shape}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def analyze_correlations(self):
        """
        Analyser les corrélations entre features - Focus xG redondance
        """
        self.logger.info("\n🔍 ANALYSE DES CORRÉLATIONS (Focus xG)")
        self.logger.info("="*60)
        
        # DataFrame pour corrélations
        df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
        
        # Matrices de corrélation (Pearson ET Spearman)
        corr_pearson = df_train.corr(method='pearson')
        corr_spearman = df_train.corr(method='spearman')
        
        # Identifier paires fortement corrélées
        high_corr_pairs = []
        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                pearson_val = abs(corr_pearson.iloc[i, j])
                spearman_val = abs(corr_spearman.iloc[i, j])
                
                # Utiliser la plus forte corrélation
                max_corr = max(pearson_val, spearman_val)
                
                if max_corr > self.corr_threshold:
                    high_corr_pairs.append({
                        'feature1': self.feature_names[i],
                        'feature2': self.feature_names[j],
                        'pearson': corr_pearson.iloc[i, j],
                        'spearman': corr_spearman.iloc[i, j],
                        'max_abs_corr': max_corr
                    })
        
        # Trier par corrélation maximale
        high_corr_pairs = sorted(high_corr_pairs, 
                                key=lambda x: x['max_abs_corr'], 
                                reverse=True)
        
        self.logger.info(f"\n🔴 {len(high_corr_pairs)} paires avec corrélation > {self.corr_threshold}:")
        self.logger.info("-"*80)
        for i, pair in enumerate(high_corr_pairs, 1):
            self.logger.info(f"{i:2}. {pair['feature1']:25} <-> {pair['feature2']:25}")
            self.logger.info(f"    Pearson: {pair['pearson']:6.3f} | Spearman: {pair['spearman']:6.3f}")
        
        # Focus sur les groupes redondants xG
        self._analyze_xg_redundancy(df_train)
        
        # Identifier features à supprimer
        features_to_remove = self._identify_redundant_features(high_corr_pairs, df_train)
        
        self.results['correlation_analysis'] = {
            'high_corr_pairs': high_corr_pairs,
            'features_to_remove': list(features_to_remove),
            'pearson_matrix': corr_pearson,
            'spearman_matrix': corr_spearman
        }
        
        self.logger.info(f"\n🗑️ Features à supprimer ({len(features_to_remove)}):")
        for feat in features_to_remove:
            self.logger.info(f"  - {feat}")
        
        return features_to_remove
    
    def _analyze_xg_redundancy(self, df_train):
        """
        Analyse spécifique de la redondance xG
        """
        self.logger.info("\n⚽ ANALYSE SPÉCIFIQUE xG REDONDANCE")
        self.logger.info("-"*50)
        
        # Groupes xG identifiés
        xg_groups = {
            'Home xG Rolling': [f for f in self.feature_names if 'home_xg_roll' in f],
            'Away xG Rolling': [f for f in self.feature_names if 'away_xg_roll' in f],
            'Home Goals Sum': [f for f in self.feature_names if 'home_goals_sum' in f],
            'Away Goals Sum': [f for f in self.feature_names if 'away_goals_sum' in f],
            'Home xG Sum': [f for f in self.feature_names if 'home_xg_sum' in f],
            'Away xG Sum': [f for f in self.feature_names if 'away_xg_sum' in f],
            'xG Efficiency': [f for f in self.feature_names if 'xg_eff' in f],
            'xG Differences': [f for f in self.feature_names if 'xg_roll' in f and 'diff' in f]
        }
        
        # Analyser chaque groupe
        for group_name, features in xg_groups.items():
            if len(features) > 1:
                group_corr = df_train[features].corr(method='pearson')
                # Moyenne corrélations hors diagonale
                n = len(features)
                avg_corr = (group_corr.abs().sum().sum() - n) / (n * (n - 1))
                
                self.logger.info(f"\n{group_name}:")
                self.logger.info(f"  Features ({len(features)}): {features}")
                self.logger.info(f"  Corrélation moyenne: {avg_corr:.3f}")
                
                if avg_corr > 0.85:
                    self.logger.info("  🔴 TRÈS REDONDANT - Garder 1 feature max")
                elif avg_corr > 0.70:
                    self.logger.info("  🟡 REDONDANT - Garder 1-2 features max")
                else:
                    self.logger.info("  ✅ Acceptable")
    
    def _identify_redundant_features(self, high_corr_pairs, df_train):
        """
        Stratégie intelligente pour supprimer redondances
        """
        # Calculer importance de chaque feature (MI avec target)
        mi_scores = mutual_info_classif(self.X_train, self.y_train, random_state=42)
        importance_scores = dict(zip(self.feature_names, mi_scores))
        
        features_to_remove = set()
        
        # Stratégie: Dans chaque paire corrélée, supprimer la moins informative
        for pair in high_corr_pairs:
            f1, f2 = pair['feature1'], pair['feature2']
            
            # Si une des deux est déjà marquée pour suppression, passer
            if f1 in features_to_remove or f2 in features_to_remove:
                continue
            
            # Comparer importance
            if importance_scores[f1] < importance_scores[f2]:
                features_to_remove.add(f1)
            else:
                features_to_remove.add(f2)
        
        # Règles spéciales pour xG (basé sur analyse Claude Opus)
        xg_special_rules = self._apply_xg_special_rules()
        features_to_remove.update(xg_special_rules)
        
        return features_to_remove
    
    def _apply_xg_special_rules(self):
        """
        Règles spéciales pour nettoyer les features xG
        """
        to_remove = set()
        
        # Règle 1: Si on a roll_5 et roll_10, garder seulement roll_5
        if 'home_xg_roll_5' in self.feature_names and 'home_xg_roll_10' in self.feature_names:
            to_remove.add('home_xg_roll_10')
        if 'away_xg_roll_5' in self.feature_names and 'away_xg_roll_10' in self.feature_names:
            to_remove.add('away_xg_roll_10')
        
        # Règle 2: Si on a sum_5 et sum_10, garder seulement sum_5
        if 'home_xg_sum_5' in self.feature_names and 'home_xg_sum_10' in self.feature_names:
            to_remove.add('home_xg_sum_10')
        if 'away_xg_sum_5' in self.feature_names and 'away_xg_sum_10' in self.feature_names:
            to_remove.add('away_xg_sum_10')
        
        # Règle 3: Si on a diff et diff_normalized, garder seulement normalized
        if 'xg_roll_5_diff' in self.feature_names and 'xg_roll_5_diff_normalized' in self.feature_names:
            to_remove.add('xg_roll_5_diff')
        if 'xg_roll_10_diff' in self.feature_names and 'xg_roll_10_diff_normalized' in self.feature_names:
            to_remove.add('xg_roll_10_diff')
        
        return to_remove
    
    def analyze_mutual_information(self):
        """
        Information mutuelle avec la target (H/D/A)
        """
        self.logger.info("\n🎯 ANALYSE D'INFORMATION MUTUELLE")
        self.logger.info("="*60)
        
        # Calculer MI scores
        mi_scores = mutual_info_classif(self.X_train, self.y_train, random_state=42)
        
        # DataFrame pour analyse
        mi_df = pd.DataFrame({
            'feature': self.feature_names,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        self.logger.info(f"\n🏆 TOP 15 FEATURES par Information Mutuelle:")
        self.logger.info("-"*70)
        for i, (_, row) in enumerate(mi_df.head(15).iterrows(), 1):
            self.logger.info(f"{i:2}. {row['feature']:35} : {row['mi_score']:.6f}")
        
        # Identifier les features avec MI très faible
        low_mi_threshold = 0.001
        low_mi_features = mi_df[mi_df['mi_score'] < low_mi_threshold]['feature'].tolist()
        
        if low_mi_features:
            self.logger.info(f"\n⚠️ Features avec MI < {low_mi_threshold}:")
            for feat in low_mi_features:
                mi_val = mi_df[mi_df['feature'] == feat]['mi_score'].values[0]
                self.logger.info(f"  - {feat}: {mi_val:.6f}")
        
        self.results['mutual_information'] = {
            'mi_df': mi_df,
            'low_mi_features': low_mi_features
        }
        
        return mi_df
    
    def analyze_variance(self):
        """
        Identifier features avec peu de variance (peu informatives)
        """
        self.logger.info("\n📊 ANALYSE DE VARIANCE")
        self.logger.info("="*60)
        
        # Standardiser pour comparer les variances
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        
        # Calculer variances
        variances = np.var(X_scaled, axis=0)
        
        # DataFrame pour analyse
        var_df = pd.DataFrame({
            'feature': self.feature_names,
            'variance': variances
        }).sort_values('variance')
        
        # Features avec faible variance
        low_var_features = var_df[var_df['variance'] < self.var_threshold]['feature'].tolist()
        
        self.logger.info(f"\n📊 VARIANCE DES FEATURES (standardisées):")
        self.logger.info("-"*60)
        for _, row in var_df.iterrows():
            status = "🔴" if row['variance'] < self.var_threshold else "✅"
            self.logger.info(f"{status} {row['feature']:35} : {row['variance']:.6f}")
        
        if low_var_features:
            self.logger.info(f"\n🔴 {len(low_var_features)} features avec variance < {self.var_threshold}")
        
        self.results['variance_analysis'] = {
            'variance_df': var_df,
            'low_variance_features': low_var_features
        }
        
        return low_var_features
    
    def select_optimal_features(self, target_n_features=12):
        """
        Sélection finale des features optimales
        """
        self.logger.info(f"\n✨ SÉLECTION DE {target_n_features} FEATURES OPTIMALES")
        self.logger.info("="*60)
        
        # Étape 1: Éliminer redondances et faible variance
        redundant = self.results['correlation_analysis']['features_to_remove']
        low_var = self.results['variance_analysis']['low_variance_features']
        low_mi = self.results['mutual_information']['low_mi_features']
        
        to_exclude = set(redundant) | set(low_var) | set(low_mi)
        features_stage1 = [f for f in self.feature_names if f not in to_exclude]
        
        self.logger.info(f"\n📌 Après nettoyage automatique: {len(features_stage1)} features")
        self.logger.info(f"   Supprimées: {len(to_exclude)} ({len(redundant)} redondantes + {len(low_var)} faible variance + {len(low_mi)} faible MI)")
        
        # Étape 2: Forcer inclusion des features critiques v2.1
        critical_features = [
            'elo_diff_normalized',
            'form_diff_normalized', 
            'h2h_score',
            'market_entropy_norm',
            'matchday_normalized'
        ]
        
        # Ajouter les features critiques si elles existent et ne sont pas déjà incluses
        for feat in critical_features:
            if feat in self.feature_names and feat not in features_stage1:
                features_stage1.append(feat)
                self.logger.info(f"   ➕ Ajout forcé feature critique: {feat}")
        
        # Étape 3: Si on a encore trop de features, utiliser RFE
        if len(features_stage1) > target_n_features:
            self.logger.info(f"\n🎯 RFE pour réduire {len(features_stage1)} → {target_n_features}")
            
            X_filtered = self.X_train[:, [self.feature_names.index(f) for f in features_stage1]]
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
            # RFE standard pour le nombre exact demandé
            rfe = RFE(rf, n_features_to_select=target_n_features, step=1)
            rfe.fit(X_filtered, self.y_train)
            
            selected_features = [features_stage1[i] for i, selected in 
                               enumerate(rfe.support_) if selected]
            
            # Forcer inclusion des features critiques même après RFE
            for feat in critical_features:
                if feat in self.feature_names and feat not in selected_features:
                    # Remplacer la feature la moins importante
                    if len(selected_features) >= target_n_features:
                        # Enlever la dernière feature selon ranking RFE
                        worst_idx = np.argmax(rfe.ranking_)
                        worst_feature = features_stage1[worst_idx]
                        if worst_feature in selected_features:
                            selected_features.remove(worst_feature)
                    selected_features.append(feat)
                    self.logger.info(f"   ➕ Inclusion forcée post-RFE: {feat}")
        else:
            selected_features = features_stage1
        
        # Affichage final
        self.logger.info(f"\n🏆 FEATURES FINALES SÉLECTIONNÉES ({len(selected_features)}):")
        self.logger.info("-"*70)
        
        # Récupérer MI scores pour affichage
        mi_df = self.results['mutual_information']['mi_df']
        
        for i, feat in enumerate(selected_features, 1):
            mi_score = mi_df[mi_df['feature'] == feat]['mi_score'].values[0]
            is_critical = "⭐" if feat in critical_features else "  "
            self.logger.info(f"{i:2}. {feat:35} (MI: {mi_score:.6f}) {is_critical}")
        
        self.results['selected_features'] = selected_features
        return selected_features
    
    def evaluate_selection(self, selected_features):
        """
        Évaluer la performance avec features sélectionnées vs complètes
        """
        self.logger.info(f"\n📊 ÉVALUATION DE LA SÉLECTION")
        self.logger.info("="*60)
        
        # Préparer les données
        selected_indices = [self.feature_names.index(f) for f in selected_features]
        X_train_selected = self.X_train[:, selected_indices]
        X_test_selected = self.X_test[:, selected_indices]
        
        # Modèle simple pour comparaison
        rf = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Évaluation avec toutes les features
        rf.fit(self.X_train, self.y_train)
        score_all = rf.score(self.X_test, self.y_test)
        
        # Évaluation avec features sélectionnées
        rf.fit(X_train_selected, self.y_train)
        score_selected = rf.score(X_test_selected, self.y_test)
        
        # Feature importance des features sélectionnées
        importance = dict(zip(selected_features, rf.feature_importances_))
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"\n📈 COMPARAISON DE PERFORMANCE:")
        self.logger.info(f"  {len(self.feature_names)} features complètes: {score_all:.4f} ({score_all*100:.2f}%)")
        self.logger.info(f"  {len(selected_features)} features sélectionnées: {score_selected:.4f} ({score_selected*100:.2f}%)")
        self.logger.info(f"  Différence: {(score_selected - score_all)*100:+.2f}pp")
        
        # Évaluation du succès
        if score_selected >= score_all - 0.005:  # Tolérance 0.5pp
            self.logger.info("\n✅ SÉLECTION RÉUSSIE: Performance maintenue avec moins de features!")
            success = True
        else:
            self.logger.info("\n⚠️ Perte de performance - Revoir la sélection")
            success = False
        
        self.logger.info(f"\n🏆 TOP 10 IMPORTANCE (features sélectionnées):")
        for i, (feat, imp) in enumerate(sorted_importance[:10], 1):
            self.logger.info(f"  {i:2}. {feat:35} : {imp:.4f}")
        
        self.results['evaluation'] = {
            'score_all_features': score_all,
            'score_selected_features': score_selected,
            'performance_diff': score_selected - score_all,
            'success': success,
            'feature_importance': sorted_importance
        }
        
        return success, score_selected, score_all
    
    def generate_comprehensive_report(self):
        """
        Rapport complet et actionnable
        """
        self.logger.info(f"\n📝 RAPPORT FINAL - ANALYSE REDONDANCE v2.2")
        self.logger.info("="*80)
        
        # Résumé des analyses
        redundant = self.results['correlation_analysis']['features_to_remove']
        low_var = self.results['variance_analysis']['low_variance_features']
        low_mi = self.results['mutual_information']['low_mi_features']
        selected = self.results['selected_features']
        
        self.logger.info(f"\n📊 RÉSUMÉ QUANTITATIF:")
        self.logger.info(f"  Features originales: {len(self.feature_names)}")
        self.logger.info(f"  Features redondantes: {len(redundant)}")
        self.logger.info(f"  Features faible variance: {len(low_var)}")
        self.logger.info(f"  Features faible MI: {len(low_mi)}")
        self.logger.info(f"  Features finales: {len(selected)}")
        self.logger.info(f"  Réduction: {((len(self.feature_names) - len(selected)) / len(self.feature_names) * 100):.1f}%")
        
        # Performance
        if 'evaluation' in self.results:
            eval_results = self.results['evaluation']
            self.logger.info(f"\n📈 IMPACT PERFORMANCE:")
            self.logger.info(f"  Avant: {eval_results['score_all_features']:.4f} ({eval_results['score_all_features']*100:.2f}%)")
            self.logger.info(f"  Après: {eval_results['score_selected_features']:.4f} ({eval_results['score_selected_features']*100:.2f}%)")
            self.logger.info(f"  Différence: {eval_results['performance_diff']*100:+.2f}pp")
            status = "✅ SUCCÈS" if eval_results['success'] else "❌ ÉCHEC"
            self.logger.info(f"  Statut: {status}")
        
        # Recommandations concrètes
        self.logger.info(f"\n🎯 RECOMMANDATIONS IMMÉDIATES:")
        self.logger.info(f"  1. ✅ Utiliser les {len(selected)} features optimales identifiées")
        self.logger.info(f"  2. 🗑️ Supprimer les {len(redundant)} features redondantes xG")
        self.logger.info(f"  3. 🔬 Tester XGBoost/LightGBM sur features nettoyées")
        self.logger.info(f"  4. 🏗️ Implémenter modèle cascade spécialisé Draws")
        self.logger.info(f"  5. 🌟 Ajouter features contextuelles (météo, arbitres, fatigue)")
        
        # Prochaines étapes
        self.logger.info(f"\n🚀 PROCHAINES ÉTAPES:")
        self.logger.info(f"  Phase 1: Entraîner v2.3 avec {len(selected)} features nettoyées")
        self.logger.info(f"  Phase 2: Développer features contextuelles innovantes")
        self.logger.info(f"  Phase 3: Architecture cascade H/D/A spécialisée")
        self.logger.info(f"  Objectif: Dépasser 55% (actuellement ~54.2%)")
        
        # Sauvegarder rapport détaillé
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        
        report = {
            'timestamp': timestamp,
            'analysis_summary': {
                'n_original_features': len(self.feature_names),
                'n_redundant_features': len(redundant),
                'n_low_variance_features': len(low_var),
                'n_low_mi_features': len(low_mi),
                'n_selected_features': len(selected),
                'reduction_percentage': (len(self.feature_names) - len(selected)) / len(self.feature_names) * 100
            },
            'feature_lists': {
                'original_features': self.feature_names,
                'redundant_features': list(redundant),
                'low_variance_features': low_var,
                'low_mi_features': low_mi,
                'selected_features': selected
            },
            'performance_impact': self.results.get('evaluation', {}),
            'high_correlation_pairs': self.results['correlation_analysis']['high_corr_pairs'][:20],
            'recommendations': {
                'immediate': [
                    f"Use {len(selected)} optimal features instead of {len(self.feature_names)}",
                    "Remove redundant xG features",
                    "Test XGBoost/LightGBM on cleaned features",
                    "Implement cascade model for Draws",
                    "Add contextual features (weather, referees, fatigue)"
                ],
                'next_phases': [
                    "Train v2.3 with cleaned features",
                    "Develop innovative contextual features", 
                    "Implement H/D/A specialized cascade architecture"
                ]
            }
        }
        
        output_file = f'reports/feature_redundancy_analysis_{timestamp}.json'
        os.makedirs('reports', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"\n✅ RAPPORT DÉTAILLÉ SAUVEGARDÉ: {output_file}")
        
        return report

def main():
    """
    Script principal d'analyse de redondance
    """
    print("🚀 ANALYSE DE REDONDANCE DES FEATURES - ODDSY v2.2")
    print("="*80)
    print("Objectif: 27 features → 10-12 features optimales")
    print("Base: Analyse critique Claude Opus")
    print("="*80)
    
    # Initialiser analyseur
    analyzer = FeatureRedundancyAnalyzer(
        correlation_threshold=0.80,  # Plus permissif pour première analyse
        variance_threshold=0.01      # Seuil variance standardisée
    )
    
    try:
        # 1. Charger données
        X_train, y_train, X_test, y_test = analyzer.load_data()
        
        # 2. Analyses principales
        redundant_features = analyzer.analyze_correlations()
        low_var_features = analyzer.analyze_variance() 
        mi_df = analyzer.analyze_mutual_information()
        
        # 3. Sélection optimale
        selected_features = analyzer.select_optimal_features(target_n_features=12)
        
        # 4. Évaluation
        success, score_selected, score_all = analyzer.evaluate_selection(selected_features)
        
        # 5. Rapport final
        report = analyzer.generate_comprehensive_report()
        
        # 6. Status final
        print("\n" + "="*80)
        print("🎯 ANALYSE TERMINÉE!")
        print(f"✅ Features optimales: {len(selected_features)}/{len(analyzer.feature_names)}")
        print(f"📊 Performance maintenue: {success}")
        print(f"📈 Différence: {(score_selected - score_all)*100:+.2f}pp")
        print("🚀 Prêt pour v2.3 avec features nettoyées!")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())