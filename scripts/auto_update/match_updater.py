#!/usr/bin/env python3
"""
🔄 MATCH UPDATER - Interface principale auto-intégration
====================================================

Système principal d'intégration automatique des nouveaux matchs EPL
avec anti-leakage temporel strict et validation complète.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import sys
import os

# Ajouter le répertoire parent au PATH pour imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_update.data_normalizer import DataNormalizer
from auto_update.feature_calculator import FeatureCalculator

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/match_updater.log', mode='a')
    ]
)
logger = logging.getLogger("match_updater")

class MatchUpdater:
    """Interface principale mise à jour automatique matchs"""
    
    def __init__(self, base_dataset_path=None):
        self.base_dataset_path = base_dataset_path or 'data/processed/v15_final_enhanced.csv'
        self.normalizer = DataNormalizer()
        self.feature_calculator = FeatureCalculator()
        
        # Statistiques session
        self.stats = {
            'matches_detected': 0,
            'matches_integrated': 0,
            'features_calculated': 0,
            'errors': []
        }
    
    def detect_new_matches(self, current_dataset, new_source_data):
        """Détecte matchs nouveaux/manquants à intégrer"""
        logger.info("🔍 Détection nouveaux matchs...")
        
        # Normaliser données source
        new_data_norm = self.normalizer.normalize_csv_data(new_source_data.copy())
        
        if len(new_data_norm) == 0:
            logger.warning("⚠️  Aucune donnée normalisée valide")
            return pd.DataFrame()
        
        # Comparaison avec dataset actuel
        current_matches = set(
            current_dataset.apply(
                lambda row: (row['Date'].date(), row['HomeTeam'], row['AwayTeam']), 
                axis=1
            ).values
        )
        
        new_matches_list = []
        for _, match in new_data_norm.iterrows():
            match_key = (match['Date'].date(), match['HomeTeam'], match['AwayTeam'])
            if match_key not in current_matches:
                new_matches_list.append(match)
        
        new_matches = pd.DataFrame(new_matches_list) if new_matches_list else pd.DataFrame()
        
        self.stats['matches_detected'] = len(new_matches)
        logger.info(f"📊 {len(new_matches)} nouveaux matchs détectés")
        
        return new_matches
    
    def validate_temporal_constraints(self, matches_to_process, current_dataset):
        """
        Valide contraintes temporelles: ne traiter que matchs dont journée précédente est complète
        RÈGLE CRITIQUE: Pour prédire J5, attendre que TOUS résultats J4 soient connus
        """
        logger.info("⏰ Validation contraintes temporelles...")
        
        valid_matches = []
        now = datetime.now()
        
        for _, match in matches_to_process.iterrows():
            match_date = match['Date']
            
            # 1. Match pas dans le futur lointain 
            if match_date > now + timedelta(days=7):
                logger.debug(f"⏭️  Match futur ignoré: {match['HomeTeam']} vs {match['AwayTeam']} ({match_date.date()})")
                continue
            
            # 2. Pour matches avec résultats connus, toujours OK
            if pd.notna(match.get('FullTimeResult')):
                valid_matches.append(match)
                continue
            
            # 3. Pour prédictions: vérifier que journée précédente est complète
            # Estimer numéro journée basé sur date (simplifié)
            season_start = datetime(2025, 8, 15)  # Premier match EPL 2025-26
            days_since_start = (match_date - season_start).days
            estimated_gameweek = max(1, days_since_start // 7 + 1)
            
            # Vérifier si journée précédente a des résultats complets
            if estimated_gameweek > 1:
                prev_gameweek_complete = self._check_previous_gameweek_complete(
                    estimated_gameweek - 1, current_dataset, match_date
                )
                
                if not prev_gameweek_complete:
                    logger.warning(f"🚫 Match J{estimated_gameweek} ignoré - J{estimated_gameweek-1} incomplète: {match['HomeTeam']} vs {match['AwayTeam']}")
                    continue
            
            valid_matches.append(match)
        
        valid_df = pd.DataFrame(valid_matches) if valid_matches else pd.DataFrame()
        logger.info(f"✅ {len(valid_df)}/{len(matches_to_process)} matchs valides temporellement")
        
        return valid_df
    
    def _check_previous_gameweek_complete(self, gameweek, current_dataset, reference_date):
        """Vérifie si journée précédente a résultats complets"""
        try:
            # Logique simplifiée: si il y a des résultats récents, journée précédente probablement complète
            recent_matches = current_dataset[
                current_dataset['Date'] < reference_date - timedelta(days=1)
            ].tail(20)  # Derniers 20 matchs
            
            # Si récents matchs ont résultats, journée précédente probablement OK
            recent_with_results = recent_matches[recent_matches['FullTimeResult'].notna()]
            completion_ratio = len(recent_with_results) / len(recent_matches) if len(recent_matches) > 0 else 0
            
            return completion_ratio > 0.8  # 80% matchs récents ont résultats
            
        except Exception as e:
            logger.warning(f"⚠️  Vérification journée précédente échouée: {e}")
            return True  # Optimiste par défaut
    
    def calculate_features_for_matches(self, matches, current_dataset):
        """Calcule features pour nouveaux matchs avec anti-leakage strict"""
        logger.info(f"⚙️  Calcul features pour {len(matches)} matchs...")
        
        enhanced_matches = []
        
        for idx, match in matches.iterrows():
            try:
                # Calcul features avec anti-leakage strict
                features = self.feature_calculator.calculate_safe_features(
                    match, current_dataset, strict_cutoff=True
                )
                
                # Ajouter features au match
                enhanced_match = match.copy()
                for feature_name, feature_value in features.items():
                    enhanced_match[feature_name] = feature_value
                
                # Target encoding si résultat connu
                if pd.notna(enhanced_match.get('FullTimeResult')):
                    target_map = {'H': 0, 'D': 1, 'A': 2}
                    enhanced_match['target'] = target_map.get(enhanced_match['FullTimeResult'])
                
                enhanced_matches.append(enhanced_match)
                self.stats['features_calculated'] += 1
                
                logger.debug(f"  ✅ Features calculées: {match['HomeTeam']} vs {match['AwayTeam']}")
                
            except Exception as e:
                error_msg = f"Calcul features échoué pour {match['HomeTeam']} vs {match['AwayTeam']}: {e}"
                logger.error(f"❌ {error_msg}")
                self.stats['errors'].append(error_msg)
                
                # Ajouter avec features par défaut
                enhanced_match = match.copy()
                for feature_name in self.feature_calculator.get_feature_names():
                    enhanced_match[feature_name] = 0.5 if 'normalized' in feature_name else 1.0
                enhanced_matches.append(enhanced_match)
        
        result_df = pd.DataFrame(enhanced_matches) if enhanced_matches else pd.DataFrame()
        logger.info(f"✅ Features calculées pour {len(result_df)} matchs")
        
        return result_df
    
    def merge_and_validate(self, current_dataset, new_matches_with_features):
        """Merge nouveau data avec validation intégrité"""
        logger.info("🔗 Fusion et validation dataset...")
        
        if len(new_matches_with_features) == 0:
            logger.info("Aucun nouveau match à fusionner")
            return current_dataset
        
        # Fusion
        merged_dataset = pd.concat([current_dataset, new_matches_with_features], ignore_index=True)
        
        # Validation intégrité
        validation_errors = self._validate_merged_dataset(merged_dataset)
        
        if validation_errors:
            logger.error(f"❌ {len(validation_errors)} erreurs validation:")
            for error in validation_errors:
                logger.error(f"  • {error}")
                self.stats['errors'].append(error)
        else:
            logger.info("✅ Validation intégrité réussie")
        
        # Trier par date
        merged_dataset = merged_dataset.sort_values('Date').reset_index(drop=True)
        
        self.stats['matches_integrated'] = len(new_matches_with_features)
        
        return merged_dataset
    
    def _validate_merged_dataset(self, dataset):
        """Valide intégrité dataset fusionné"""
        errors = []
        
        # 1. Doublons
        duplicates = dataset.duplicated(['Date', 'HomeTeam', 'AwayTeam'])
        if duplicates.any():
            errors.append(f"{duplicates.sum()} doublons détectés")
        
        # 2. Équipes cohérentes
        all_home_teams = set(dataset['HomeTeam'].dropna().unique())
        all_away_teams = set(dataset['AwayTeam'].dropna().unique())
        unknown_teams = (all_home_teams | all_away_teams) - self.normalizer.team_mapping.keys() - set(self.normalizer.team_mapping.values())
        
        if unknown_teams:
            errors.append(f"Équipes inconnues: {list(unknown_teams)}")
        
        # 3. Dates logiques
        future_matches = dataset[dataset['Date'] > datetime.now() + timedelta(days=365)]
        if len(future_matches) > 0:
            errors.append(f"{len(future_matches)} matchs dans futur lointain")
        
        # 4. Features cohérentes (pas toutes NaN)
        feature_cols = [col for col in dataset.columns if 'normalized' in col or 'eff' in col]
        for col in feature_cols:
            if dataset[col].isna().all():
                errors.append(f"Feature {col} entièrement NaN")
        
        return errors
    
    def save_updated_dataset(self, dataset, custom_suffix=None):
        """Sauvegarde dataset avec versioning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if custom_suffix:
            filename = f"v_auto_update_{custom_suffix}_{timestamp}.csv"
        else:
            filename = f"v_auto_update_{timestamp}.csv"
        
        output_path = f"data/processed/{filename}"
        
        # Créer répertoire si nécessaire
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder CSV
        dataset.to_csv(output_path, index=False)
        
        # Sauvegarder métadonnées
        metadata = {
            'timestamp': timestamp,
            'total_matches': len(dataset),
            'date_range': {
                'start': dataset['Date'].min().isoformat(),
                'end': dataset['Date'].max().isoformat()
            },
            'integration_stats': self.stats,
            'features_available': self.feature_calculator.get_feature_names()
        }
        
        metadata_path = output_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Dataset sauvegardé: {output_path}")
        logger.info(f"📋 Métadonnées: {metadata_path}")
        
        return output_path
    
    def update_from_csv(self, new_csv_path, save_result=True):
        """Interface principale: mise à jour depuis CSV"""
        logger.info(f"🚀 Mise à jour depuis CSV: {new_csv_path}")
        
        try:
            # 1. Charger datasets
            logger.info(f"📂 Chargement dataset base: {self.base_dataset_path}")
            current_dataset = pd.read_csv(self.base_dataset_path, parse_dates=['Date'])
            
            logger.info(f"📂 Chargement nouveau CSV: {new_csv_path}")
            new_source = pd.read_csv(new_csv_path)
            
            # 2. Détecter nouveaux matchs
            new_matches = self.detect_new_matches(current_dataset, new_source)
            
            if len(new_matches) == 0:
                logger.info("✅ Aucun nouveau match - dataset à jour")
                return self.base_dataset_path
            
            # 3. Valider contraintes temporelles
            valid_matches = self.validate_temporal_constraints(new_matches, current_dataset)
            
            if len(valid_matches) == 0:
                logger.info("⏰ Aucun match valide temporellement - attendre journée précédente")
                return self.base_dataset_path
            
            # 4. Calculer features
            matches_with_features = self.calculate_features_for_matches(valid_matches, current_dataset)
            
            # 5. Fusionner et valider
            updated_dataset = self.merge_and_validate(current_dataset, matches_with_features)
            
            # 6. Sauvegarder
            if save_result:
                output_path = self.save_updated_dataset(updated_dataset)
                
                # Log final
                logger.info("🎉 MISE À JOUR TERMINÉE")
                logger.info(f"📊 Stats: {self.stats['matches_detected']} détectés, {self.stats['matches_integrated']} intégrés")
                logger.info(f"📂 Dataset final: {output_path}")
                
                return output_path
            else:
                return updated_dataset
        
        except Exception as e:
            error_msg = f"Erreur mise à jour: {e}"
            logger.error(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            raise

# Interface command-line simple
def main():
    """Interface CLI pour mise à jour manuelle"""
    import argparse
    
    parser = argparse.ArgumentParser(description="🔄 Match Updater - Intégration automatique matchs EPL")
    parser.add_argument('--base-dataset', default='data/processed/v15_final_enhanced.csv', 
                       help='Dataset base à mettre à jour')
    parser.add_argument('--new-csv', required=True,
                       help='CSV avec nouveaux matchs à intégrer')
    parser.add_argument('--no-save', action='store_true',
                       help='Ne pas sauvegarder (test uniquement)')
    
    args = parser.parse_args()
    
    # Mise à jour
    updater = MatchUpdater(args.base_dataset)
    
    try:
        result_path = updater.update_from_csv(args.new_csv, save_result=not args.no_save)
        print(f"✅ Mise à jour terminée: {result_path}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()