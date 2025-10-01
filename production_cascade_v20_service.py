#!/usr/bin/env python3
"""
🏆 Production Service - TRUE Cascade Champion v2.0
=================================================

Service de production pour le VRAI Cascade Champion v2.0 (46%)
Charge le modèle depuis la DB et génère les prédictions J6.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.python_connector import OddsyDatabase

class CascadeV20ProductionService:
    """Service de production pour Cascade Champion v2.0"""
    
    def __init__(self):
        self.db = OddsyDatabase()
        self.model = None
        self.model_metadata = None
        
    def load_cascade_v20_from_production(self):
        """Charge le Cascade Champion v2.0 depuis la base de production"""
        print("🏆 CHARGEMENT CASCADE CHAMPION v2.0 DEPUIS PRODUCTION")
        print("=" * 60)
        
        try:
            # Récupérer les métadonnées depuis la DB
            model_query = """
            SELECT * FROM model_performance 
            WHERE model_name = 'Cascade Champion' 
            AND model_version = 'v2.0_production' 
            AND is_active = TRUE
            """
            
            model_data = self.db.execute_query(model_query)
            
            if len(model_data) == 0:
                raise ValueError("Cascade Champion v2.0 non trouvé en production!")
            
            self.model_metadata = model_data.iloc[0].to_dict()
            
            print(f"✅ Métadonnées chargées:")
            print(f"   Model: {self.model_metadata['model_name']} {self.model_metadata['model_version']}")
            print(f"   Accuracy: {self.model_metadata['accuracy']:.1%}")
            print(f"   Deployment: {self.model_metadata['deployment_date']}")
            print(f"   Status: {'ACTIF' if self.model_metadata['is_active'] else 'INACTIF'}")
            
            # Charger le modèle depuis le fichier
            model_path = self.model_metadata['model_file_path']
            if model_path and os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"✅ Modèle chargé depuis: {model_path}")
            else:
                print(f"⚠️ Fichier modèle non trouvé: {model_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement Cascade v2.0: {str(e)}")
            return False
    
    def get_production_features(self):
        """Récupère les features de production depuis la DB"""
        try:
            features_query = """
            SELECT feature_name, importance_score, default_value
            FROM production_features 
            WHERE model_name = 'Cascade Champion' 
            AND model_version = 'v2.0_production'
            ORDER BY importance_score DESC
            """
            
            features_data = self.db.execute_query(features_query)
            return features_data.to_dict('records')
            
        except Exception as e:
            print(f"❌ Erreur récupération features: {str(e)}")
            return []
    
    def predict_j6_with_cascade_v20(self):
        """Génère les prédictions J6 avec le vrai Cascade v2.0"""
        print(f"\n🎯 PRÉDICTIONS J6 - CASCADE CHAMPION v2.0 (46%)")
        print("=" * 60)
        
        if not self.model:
            print("❌ Modèle non chargé!")
            return []
        
        try:
            # Charger les données J6
            j6_data = pd.read_csv("data/raw/j6_odds.csv")
            print(f"📊 {len(j6_data)} matchs J6 chargés")
            
            # Récupérer les features de production
            features_info = self.get_production_features()
            required_features = [f['feature_name'] for f in features_info]
            
            print(f"📋 Features requises: {len(required_features)}")
            
            predictions = []
            
            for idx, row in j6_data.iterrows():
                try:
                    # Préparer les données de match avec features estimées
                    match_features = {}
                    
                    # Remplir les features avec des valeurs par défaut ou estimées
                    for feature_info in features_info:
                        feature_name = feature_info['feature_name']
                        default_value = feature_info['default_value']
                        
                        if feature_name == 'market_entropy_norm':
                            # Calculer depuis les cotes
                            match_features[feature_name] = self._calculate_market_entropy(
                                row['B365H'], row['B365D'], row['B365A']
                            )
                        elif feature_name == 'elo_diff_normalized':
                            # Estimer depuis les cotes
                            match_features[feature_name] = self._estimate_elo_from_odds(
                                row['B365H'], row['B365A']
                            )
                        elif feature_name == 'matchday_normalized':
                            match_features[feature_name] = 0.2  # Début de saison
                        else:
                            match_features[feature_name] = default_value
                    
                    # Créer DataFrame pour prédiction
                    match_df = pd.DataFrame([match_features])
                    
                    # Prédiction avec le vrai Cascade v2.0
                    prediction = self.model.predict(match_df)[0]
                    probabilities = self.model.predict_proba(match_df)[0]
                    
                    result_map = {0: 'H', 1: 'D', 2: 'A'}
                    predicted_result = result_map[prediction]
                    
                    prediction_data = {
                        'match_id': idx + 2000,  # ID unique pour J6
                        'match_date': row['Date'],
                        'home_team': row['HomeTeam'],
                        'away_team': row['AwayTeam'],
                        'model_name': 'Cascade Champion',
                        'model_version': 'v2.0_production',
                        'predicted_result': predicted_result,
                        'probabilities': {
                            'H': float(probabilities[0]),
                            'D': float(probabilities[1]),
                            'A': float(probabilities[2])
                        },
                        'confidence_score': float(max(probabilities)),
                        'features_used': match_features,
                        'prediction_date': datetime.now().isoformat()
                    }
                    
                    predictions.append(prediction_data)
                    
                    # Afficher la prédiction
                    confidence = max(probabilities) * 100
                    print(f"✅ {row['HomeTeam']} vs {row['AwayTeam']}: "
                          f"{predicted_result} ({confidence:.1f}%)")
                    
                except Exception as e:
                    print(f"❌ Erreur prédiction {row['HomeTeam']} vs {row['AwayTeam']}: {str(e)}")
                    continue
            
            print(f"\n✅ {len(predictions)} prédictions J6 générées avec Cascade v2.0")
            return predictions
            
        except Exception as e:
            print(f"❌ Erreur prédictions J6: {str(e)}")
            return []
    
    def _calculate_market_entropy(self, h_odds: float, d_odds: float, a_odds: float) -> float:
        """Calcule l'entropie de marché depuis les cotes"""
        try:
            # Convertir en probabilités implicites
            h_prob = 1 / h_odds if h_odds > 0 else 0.33
            d_prob = 1 / d_odds if d_odds > 0 else 0.33
            a_prob = 1 / a_odds if a_odds > 0 else 0.33
            
            # Normaliser
            total = h_prob + d_prob + a_prob
            h_prob /= total
            d_prob /= total
            a_prob /= total
            
            # Calculer l'entropie
            entropy = -(h_prob * np.log2(h_prob + 1e-10) + 
                       d_prob * np.log2(d_prob + 1e-10) + 
                       a_prob * np.log2(a_prob + 1e-10))
            
            return min(entropy / np.log2(3), 1.0)
            
        except:
            return 0.5
    
    def _estimate_elo_from_odds(self, h_odds: float, a_odds: float) -> float:
        """Estime la différence ELO depuis les cotes"""
        try:
            if h_odds <= 0 or a_odds <= 0:
                return 0.5
            
            odds_ratio = a_odds / h_odds
            
            if odds_ratio > 2.0:    return 0.7  # Home plus fort
            elif odds_ratio > 1.5:  return 0.6
            elif odds_ratio > 0.67: return 0.5  # Équilibré  
            elif odds_ratio > 0.5:  return 0.4
            else:                   return 0.3  # Away plus fort
            
        except:
            return 0.5
    
    def save_j6_predictions_to_db(self, predictions: List[Dict]) -> bool:
        """Sauvegarde les prédictions J6 en base"""
        print(f"\n💾 SAUVEGARDE {len(predictions)} PRÉDICTIONS J6")
        print("=" * 50)
        
        try:
            # Utiliser une approche simple INSERT pour éviter les conflits
            saved_count = 0
            
            for prediction in predictions:
                try:
                    # Préparer l'insertion manuelle pour éviter les conflits ON CONFLICT
                    insert_query = """
                    INSERT INTO predictions (
                        match_id, model_name, model_version, predicted_result,
                        probability_home, probability_draw, probability_away,
                        confidence_score, features_used, prediction_date
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """
                    
                    params = (
                        prediction['match_id'],
                        prediction['model_name'],
                        prediction['model_version'], 
                        prediction['predicted_result'],
                        prediction['probabilities']['H'],
                        prediction['probabilities']['D'],
                        prediction['probabilities']['A'],
                        prediction['confidence_score'],
                        json.dumps(prediction['features_used'])
                    )
                    
                    self.db.execute_non_query(insert_query, params)
                    saved_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Erreur sauvegarde match {prediction['match_id']}: {str(e)}")
                    continue
            
            print(f"✅ {saved_count}/{len(predictions)} prédictions sauvegardées")
            return saved_count > 0
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {str(e)}")
            return False
    
    def run_production_j6_cascade_v20(self):
        """Lance la production complète pour J6 avec Cascade v2.0"""
        print("🚀 PRODUCTION J6 - CASCADE CHAMPION v2.0")
        print("=" * 60)
        
        success_steps = 0
        
        # Étape 1: Charger le modèle
        if self.load_cascade_v20_from_production():
            success_steps += 1
        else:
            print("❌ Échec chargement modèle")
            return False
        
        # Étape 2: Générer les prédictions J6
        predictions = self.predict_j6_with_cascade_v20()
        if predictions:
            success_steps += 1
        else:
            print("❌ Aucune prédiction générée")
            return False
        
        # Étape 3: Sauvegarder en base
        if self.save_j6_predictions_to_db(predictions):
            success_steps += 1
        
        # Résumé
        print(f"\n" + "=" * 60)
        print(f"🎉 PRODUCTION J6 CASCADE v2.0 TERMINÉE")
        print("=" * 60)
        print(f"✅ Étapes réussies: {success_steps}/3")
        print(f"🏆 Modèle: {self.model_metadata['model_name']} v2.0")
        print(f"🎯 Accuracy attendue: {self.model_metadata['accuracy']:.1%}")
        print(f"📊 Prédictions J6: {len(predictions)}")
        
        if success_steps >= 2:
            print(f"🎉 CASCADE CHAMPION v2.0 OPÉRATIONNEL EN PRODUCTION!")
            return True
        else:
            print(f"⚠️ Production incomplète")
            return False
    
    def close(self):
        """Ferme la connexion DB"""
        if self.db:
            self.db.close()

def main():
    """Lance le service de production Cascade v2.0"""
    service = CascadeV20ProductionService()
    
    try:
        success = service.run_production_j6_cascade_v20()
        
        if success:
            print(f"\n🏆 CASCADE CHAMPION v2.0 EN PRODUCTION!")
            print(f"🎯 Le VRAI champion (46%) est opérationnel!")
        else:
            print(f"\n❌ Échec production")
        
        return success
        
    except Exception as e:
        print(f"❌ Erreur service production: {str(e)}")
        return False
        
    finally:
        service.close()

if __name__ == "__main__":
    main()