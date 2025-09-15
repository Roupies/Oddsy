#!/usr/bin/env python3
"""
Validation Rolling Predictions - VERSION CORRIGÉE
==================================================

BUG CORRIGÉ: Utilise les features de base du v2.3 au lieu des features
recalibrées du domain adaptation qui n'existent pas dans le dataset original.

MISSION: Audit rigoureux avec le bon modèle et les bonnes features.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RollingPredictionsValidatorFixed:
    """
    Validateur de prédictions rolling CORRIGÉ
    """
    
    def __init__(self):
        self.dataset_path = "data/processed/v15_final_enhanced.csv"
        self.ground_truth_path = "data/validation/ground_truth_gw1_4.csv"
        
        # CORRECTION: Utiliser le modèle v2.3 original avec features de base
        self.model_path = "models/v23_retrained_2025_09_11_154613.joblib"
        
        # Features du v2.3 (d'après CLAUDE.md)
        self.v23_features = [
            'elo_diff_normalized',
            'market_entropy_norm', 
            'shots_diff_normalized',
            'corners_diff_normalized',
            'form_diff_normalized',
            'h2h_score',
            'matchday_normalized',
            'home_xg_eff_10',
            'away_xg_eff_10',
            'away_goals_sum_5'
        ]
        
        # Données
        self.full_dataset = None
        self.ground_truth = None
        self.historical_data = None
        self.model = None
        
        # Résultats
        self.all_predictions = []
        
    def load_all_data(self):
        """Charger toutes les données nécessaires"""
        
        print("📊 CHARGEMENT DES DONNÉES - VERSION CORRIGÉE")
        print("="*60)
        
        # Dataset complet
        self.full_dataset = pd.read_csv(self.dataset_path)
        self.full_dataset['Date'] = pd.to_datetime(self.full_dataset['Date'])
        print(f"✅ Dataset: {len(self.full_dataset)} matches")
        
        # Vérité terrain GW1-4
        self.ground_truth = pd.read_csv(self.ground_truth_path)
        self.ground_truth['Date'] = pd.to_datetime(self.ground_truth['Date'], dayfirst=True)
        print(f"✅ Vérité terrain: {len(self.ground_truth)} matches")
        
        # CORRECTION: Modèle v2.3 original
        print(f"📂 Chargement modèle v2.3: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print(f"✅ Modèle v2.3 chargé: {type(self.model)}")
        
        # Données historiques (avant EPL 2025-26)
        epl_2025_start = pd.to_datetime('2025-08-15')
        self.historical_data = self.full_dataset[
            self.full_dataset['Date'] < epl_2025_start
        ].copy()
        print(f"✅ Données historiques: {len(self.historical_data)} matches")
        
        # Vérifier que toutes les features v2.3 sont disponibles
        available_features = set(self.full_dataset.columns)
        missing_features = [f for f in self.v23_features if f not in available_features]
        
        if missing_features:
            print(f"⚠️  Features manquantes: {missing_features}")
        else:
            print(f"✅ Toutes les features v2.3 disponibles ({len(self.v23_features)})")
        
        return {
            'total_dataset': len(self.full_dataset),
            'ground_truth': len(self.ground_truth),
            'historical': len(self.historical_data),
            'missing_features': missing_features
        }
        
    def prepare_features_for_match(self, historical_data, home_team, away_team, match_date):
        """
        CORRECTION: Préparer les vraies features v2.3 pour un match
        """
        
        # Filtrer l'historique pour éviter look-ahead bias
        available_history = historical_data[
            historical_data['Date'] < match_date
        ].copy()
        
        if len(available_history) == 0:
            return None
            
        # Stratégie simplifiée: prendre les dernières features disponibles pour ces équipes
        # Dans un vrai système, on recalculerait toutes les features dynamiquement
        
        # Chercher les derniers matchs de chaque équipe
        home_last_matches = available_history[
            (available_history['HomeTeam'] == home_team) | 
            (available_history['AwayTeam'] == home_team)
        ].tail(5)
        
        away_last_matches = available_history[
            (available_history['HomeTeam'] == away_team) | 
            (available_history['AwayTeam'] == away_team)
        ].tail(5)
        
        if len(home_last_matches) == 0 or len(away_last_matches) == 0:
            return None
            
        # Prendre les features du dernier match où ces équipes ont joué l'une contre l'autre
        h2h_matches = available_history[
            ((available_history['HomeTeam'] == home_team) & (available_history['AwayTeam'] == away_team)) |
            ((available_history['HomeTeam'] == away_team) & (available_history['AwayTeam'] == home_team))
        ]
        
        if len(h2h_matches) > 0:
            # Utiliser le dernier match H2H
            reference_match = h2h_matches.iloc[-1]
            
            # Ajuster les features selon qui joue à domicile
            if reference_match['HomeTeam'] == home_team:
                # Même configuration
                features_dict = {}
                for feature in self.v23_features:
                    if feature in reference_match:
                        features_dict[feature] = reference_match[feature]
                    else:
                        features_dict[feature] = 0.5  # Valeur neutre
            else:
                # Configuration inversée - inverser les features
                features_dict = {}
                for feature in self.v23_features:
                    if feature in reference_match:
                        if 'diff' in feature:
                            # Inverser les différences
                            features_dict[feature] = 1 - reference_match[feature] if not pd.isna(reference_match[feature]) else 0.5
                        else:
                            features_dict[feature] = reference_match[feature]
                    else:
                        features_dict[feature] = 0.5
        else:
            # Pas de H2H récent - utiliser des moyennes
            features_dict = {}
            for feature in self.v23_features:
                if feature in available_history.columns:
                    mean_val = available_history[feature].mean()
                    features_dict[feature] = mean_val if not pd.isna(mean_val) else 0.5
                else:
                    features_dict[feature] = 0.5
        
        # Convertir en array
        features_array = np.array([features_dict[f] for f in self.v23_features]).reshape(1, -1)
        
        return features_array
        
    def simulate_rolling_predictions_corrected(self):
        """
        CORRECTION: Simulation avec vraies features v2.3
        """
        
        print(f"\n🎯 SIMULATION ROLLING CORRIGÉE - FEATURES V2.3")
        print("="*60)
        
        ground_truth_sorted = self.ground_truth.sort_values('Date')
        current_history = self.historical_data.copy()
        
        print(f"\n🔄 PRÉDICTIONS ROLLING CORRIGÉES:")
        print("-" * 60)
        
        for idx, match in ground_truth_sorted.iterrows():
            match_date = match['Date']
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            actual_result = match['FTR']
            
            print(f"🏟️  {match_date.strftime('%d/%m')} | {home_team} vs {away_team}")
            
            # CORRECTION: Préparer vraies features v2.3
            features = self.prepare_features_for_match(current_history, home_team, away_team, match_date)
            
            if features is not None:
                # Prédiction avec modèle v2.3
                try:
                    prediction_proba = self.model.predict_proba(features)[0]
                    
                    # v2.3 est un modèle 3-classes (H/D/A)
                    if len(prediction_proba) == 3:
                        home_prob = prediction_proba[0]  # Classe 0 = Home
                        draw_prob = prediction_proba[1]  # Classe 1 = Draw  
                        away_prob = prediction_proba[2]  # Classe 2 = Away
                        
                        # Prédire la classe avec la plus haute probabilité
                        predicted_class = np.argmax(prediction_proba)
                        predicted_result = ['H', 'D', 'A'][predicted_class]
                    else:
                        # Fallback pour modèle binaire
                        home_prob = prediction_proba[1] if len(prediction_proba) == 2 else prediction_proba[0]
                        draw_prob = 0
                        away_prob = 1 - home_prob
                        predicted_result = 'H' if home_prob > 0.5 else 'A'
                    
                    # Stockage de la prédiction
                    prediction_record = {
                        'Date': match_date,
                        'HomeTeam': home_team,
                        'AwayTeam': away_team,
                        'Predicted_FTR': predicted_result,
                        'Actual_FTR': actual_result,
                        'Home_Prob': home_prob,
                        'Draw_Prob': draw_prob,
                        'Away_Prob': away_prob,
                        'Correct_Prediction': predicted_result == actual_result
                    }
                    
                    self.all_predictions.append(prediction_record)
                    
                    correct = "✅" if predicted_result == actual_result else "❌"
                    print(f"   Prédiction: {predicted_result} | Réel: {actual_result} | P(H/D/A)=({home_prob:.2f}/{draw_prob:.2f}/{away_prob:.2f}) {correct}")
                    
                except Exception as e:
                    print(f"   ⚠️  Erreur prédiction: {e}")
            else:
                print(f"   ⚠️  Pas assez d'historique pour prédire")
            
            # Ajouter ce match à l'historique pour les prédictions suivantes
            match_to_add = pd.DataFrame([match])
            current_history = pd.concat([current_history, match_to_add], ignore_index=True)
        
        print(f"\n✅ Simulation corrigée terminée: {len(self.all_predictions)} prédictions")
        
        return self.all_predictions
        
    def save_corrected_predictions(self):
        """Sauvegarder les prédictions corrigées"""
        
        if not self.all_predictions:
            print("⚠️  Aucune prédiction à sauvegarder")
            return None
            
        predictions_df = pd.DataFrame(self.all_predictions)
        
        # Statistiques
        accuracy = predictions_df['Correct_Prediction'].mean()
        total_predictions = len(predictions_df)
        
        print(f"\n📊 STATISTIQUES CORRIGÉES:")
        print(f"   Total prédictions: {total_predictions}")
        print(f"   Accuracy corrigée: {accuracy:.1%}")
        
        # Distribution des prédictions
        pred_dist = predictions_df['Predicted_FTR'].value_counts()
        actual_dist = predictions_df['Actual_FTR'].value_counts()
        
        print(f"   Prédictions → H:{pred_dist.get('H',0)}, D:{pred_dist.get('D',0)}, A:{pred_dist.get('A',0)}")
        print(f"   Réels       → H:{actual_dist.get('H',0)}, D:{actual_dist.get('D',0)}, A:{actual_dist.get('A',0)}")
        
        # Sauvegarder
        output_file = Path("data/validation/replicated_predictions_gw1_4_CORRECTED.csv")
        predictions_df.to_csv(output_file, index=False)
        
        print(f"\n💾 Prédictions corrigées sauvegardées: {output_file}")
        
        return {
            'predictions_file': output_file,
            'total_predictions': total_predictions,
            'corrected_accuracy': accuracy
        }
        
    def run_corrected_validation(self):
        """Exécuter la validation corrigée complète"""
        
        print("🔍 VALIDATION ROLLING CORRIGÉE - MODÈLE V2.3")
        print("="*70)
        
        # Charger les données
        data_info = self.load_all_data()
        
        if data_info['missing_features']:
            print(f"❌ Impossible de continuer - features manquantes")
            return None
        
        # Simuler les prédictions rolling corrigées
        predictions = self.simulate_rolling_predictions_corrected()
        
        # Sauvegarder
        save_info = self.save_corrected_predictions()
        
        print(f"\n🎉 VALIDATION CORRIGÉE TERMINÉE!")
        print(f"📊 Accuracy corrigée: {save_info['corrected_accuracy']:.1%}")
        
        return {
            'data_info': data_info,
            'predictions_count': len(predictions),
            'save_info': save_info
        }

def main():
    """Fonction principale"""
    
    validator = RollingPredictionsValidatorFixed()
    results = validator.run_corrected_validation()
    
    return validator, results

if __name__ == "__main__":
    validator, results = main()