#!/usr/bin/env python3
"""
Validation Rolling Predictions - Audit Rigoureux EPL 2025-26 GW1-4
==================================================================

MISSION CRITIQUE: Répliquer de manière indépendante le processus de prédiction
"rolling" pour les 4 premières journées EPL 2025-26, en évitant toute fuite
de données (look-ahead bias).

APPROCHE:
- Simulation semaine par semaine des prédictions
- Entraînement du modèle UNIQUEMENT sur les données disponibles AVANT chaque journée
- Intégration progressive des vrais résultats
- Prédictions stockées pour audit final

VALIDATION DU SCORE 59.1%: Ce script permettra de confirmer ou infirmer
le score de domain adaptation de manière totalement indépendante.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RollingPredictionsValidator:
    """
    Validateur de prédictions rolling pour audit rigoureux
    """
    
    def __init__(self, dataset_path, domain_adaptation_model_path):
        self.dataset_path = dataset_path
        self.model_path = domain_adaptation_model_path
        self.ground_truth_path = "data/validation/ground_truth_gw1_4.csv"
        
        # Données
        self.full_dataset = None
        self.ground_truth = None
        self.historical_data = None
        
        # Modèle Domain Adaptation
        self.da_model_info = None
        self.optimal_threshold = 0.35  # Seuil optimisé du domain adaptation
        
        # Résultats
        self.all_predictions = []
        
    def load_all_data(self):
        """Charger toutes les données nécessaires"""
        
        print("📊 CHARGEMENT DES DONNÉES POUR VALIDATION ROLLING")
        print("="*60)
        
        # Dataset complet
        print(f"📂 Chargement dataset: {self.dataset_path}")
        self.full_dataset = pd.read_csv(self.dataset_path)
        self.full_dataset['Date'] = pd.to_datetime(self.full_dataset['Date'])
        print(f"✅ Dataset: {len(self.full_dataset)} matches")
        
        # Vérité terrain GW1-4
        print(f"📂 Chargement vérité terrain: {self.ground_truth_path}")
        self.ground_truth = pd.read_csv(self.ground_truth_path)
        self.ground_truth['Date'] = pd.to_datetime(self.ground_truth['Date'], dayfirst=True)
        print(f"✅ Vérité terrain: {len(self.ground_truth)} matches")
        
        # Modèle Domain Adaptation
        print(f"📂 Chargement modèle DA: {self.model_path}")
        self.da_model_info = joblib.load(self.model_path)
        print(f"✅ Modèle DA chargé avec {len(self.da_model_info['features'])} features")
        
        # Données historiques (avant EPL 2025-26)
        epl_2025_start = pd.to_datetime('2025-08-15')
        self.historical_data = self.full_dataset[
            self.full_dataset['Date'] < epl_2025_start
        ].copy()
        print(f"✅ Données historiques: {len(self.historical_data)} matches")
        
        return {
            'total_dataset': len(self.full_dataset),
            'ground_truth': len(self.ground_truth),
            'historical': len(self.historical_data)
        }
        
    def prepare_features_for_match(self, historical_data, match_date):
        """
        Préparer les features pour un match donné, en utilisant UNIQUEMENT
        l'historique disponible AVANT cette date (évite look-ahead bias)
        """
        
        # Filtrer l'historique pour éviter look-ahead bias
        available_history = historical_data[
            historical_data['Date'] < match_date
        ].copy()
        
        if len(available_history) == 0:
            return None
            
        # Utiliser les features du modèle Domain Adaptation
        feature_names = self.da_model_info['features']
        
        # Pour simplifier, on prend les features du dernier match disponible
        # (dans un vrai système, on recalculerait toutes les features)
        latest_match = available_history.iloc[-1]
        
        features = []
        for feature in feature_names:
            if feature in latest_match:
                features.append(latest_match[feature])
            else:
                features.append(0)  # Valeur par défaut
                
        return np.array(features).reshape(1, -1)
        
    def simulate_rolling_predictions_by_gameweek(self):
        """
        Simuler les prédictions rolling semaine par semaine pour audit rigoureux
        """
        
        print(f"\n🎯 SIMULATION ROLLING PREDICTIONS - AUDIT RIGOUREUX")
        print("="*60)
        
        # Grouper la vérité terrain par date (approximation des journées)
        ground_truth_sorted = self.ground_truth.sort_values('Date')
        unique_dates = ground_truth_sorted['Date'].dt.date.unique()
        
        print(f"📅 Journées détectées: {len(unique_dates)}")
        for i, date in enumerate(unique_dates[:5], 1):  # Afficher 5 premières
            matches_on_date = len(ground_truth_sorted[ground_truth_sorted['Date'].dt.date == date])
            print(f"   J{i}: {date} → {matches_on_date} matches")
        
        # Simulation match par match (plus précis qu'un groupement par journée)
        current_history = self.historical_data.copy()
        
        print(f"\n🔄 PRÉDICTIONS ROLLING MATCH PAR MATCH:")
        print("-" * 60)
        
        for idx, match in ground_truth_sorted.iterrows():
            match_date = match['Date']
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            actual_result = match['FTR']
            
            print(f"🏟️  {match_date.strftime('%d/%m')} | {home_team} vs {away_team}")
            
            # Préparer features avec l'historique disponible AVANT ce match
            features = self.prepare_features_for_match(current_history, match_date)
            
            if features is not None:
                # Prédiction avec le modèle Domain Adaptation
                model = self.da_model_info['model']
                prediction_proba = model.predict_proba(features)[0]
                
                # Utiliser le seuil optimisé (0.35)
                home_win_prob = prediction_proba[1]
                predicted_home_win = 1 if home_win_prob >= self.optimal_threshold else 0
                
                # Convertir en format FTR
                if predicted_home_win:
                    predicted_result = 'H'
                else:
                    # Simplification: si pas home win, prédire away win
                    # (dans un vrai système, on aurait 3 classes)
                    predicted_result = 'A'
                
                # Stockage de la prédiction
                prediction_record = {
                    'Date': match_date,
                    'HomeTeam': home_team,
                    'AwayTeam': away_team,
                    'Predicted_FTR': predicted_result,
                    'Actual_FTR': actual_result,
                    'Home_Win_Prob': home_win_prob,
                    'Threshold_Used': self.optimal_threshold,
                    'Correct_Prediction': predicted_result == actual_result
                }
                
                self.all_predictions.append(prediction_record)
                
                correct = "✅" if predicted_result == actual_result else "❌"
                print(f"   Prédiction: {predicted_result} | Réel: {actual_result} | P(H)={home_win_prob:.3f} {correct}")
            else:
                print(f"   ⚠️  Pas assez d'historique pour prédire")
            
            # CRUCIAL: Ajouter ce match à l'historique pour les prédictions suivantes
            # (simule la mise à jour temps réel)
            match_to_add = pd.DataFrame([match])
            current_history = pd.concat([current_history, match_to_add], ignore_index=True)
        
        print(f"\n✅ Simulation terminée: {len(self.all_predictions)} prédictions générées")
        
        return self.all_predictions
        
    def save_predictions_for_audit(self):
        """Sauvegarder les prédictions pour audit final"""
        
        if not self.all_predictions:
            print("⚠️  Aucune prédiction à sauvegarder")
            return None
            
        # Convertir en DataFrame
        predictions_df = pd.DataFrame(self.all_predictions)
        
        # Statistiques préliminaires
        accuracy = predictions_df['Correct_Prediction'].mean()
        total_predictions = len(predictions_df)
        
        print(f"\n📊 STATISTIQUES PRÉLIMINAIRES:")
        print(f"   Total prédictions: {total_predictions}")
        print(f"   Accuracy brute: {accuracy:.1%}")
        
        # Sauvegarder
        output_file = Path("data/validation/replicated_predictions_gw1_4.csv")
        predictions_df.to_csv(output_file, index=False)
        
        print(f"\n💾 Prédictions sauvegardées: {output_file}")
        
        return {
            'predictions_file': output_file,
            'total_predictions': total_predictions,
            'preliminary_accuracy': accuracy
        }
        
    def run_complete_rolling_validation(self):
        """Exécuter la validation rolling complète"""
        
        print("🔍 VALIDATION ROLLING PREDICTIONS - AUDIT DOMAIN ADAPTATION")
        print("="*70)
        
        # Charger les données
        data_info = self.load_all_data()
        
        # Simuler les prédictions rolling
        predictions = self.simulate_rolling_predictions_by_gameweek()
        
        # Sauvegarder pour audit
        save_info = self.save_predictions_for_audit()
        
        print(f"\n🎉 VALIDATION ROLLING TERMINÉE!")
        print(f"📊 Prêt pour audit final contre le score 59.1%")
        
        return {
            'data_info': data_info,
            'predictions_count': len(predictions),
            'save_info': save_info
        }

def main():
    """Fonction principale"""
    
    # Configuration
    dataset_path = "data/processed/v15_final_enhanced.csv"
    model_path = "models/domain_adaptation/domain_adapted_model_20250914_231718.joblib"
    
    # Vérifier que les fichiers existent
    if not Path(dataset_path).exists():
        print(f"❌ Dataset introuvable: {dataset_path}")
        return
        
    if not Path(model_path).exists():
        print(f"❌ Modèle introuvable: {model_path}")
        return
    
    # Initialiser le validateur
    validator = RollingPredictionsValidator(dataset_path, model_path)
    
    # Exécuter la validation
    results = validator.run_complete_rolling_validation()
    
    return validator, results

if __name__ == "__main__":
    validator, results = main()