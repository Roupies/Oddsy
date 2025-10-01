#!/usr/bin/env python3
"""
🎯 Advanced Ensemble Strategies - Stacking & Meta-Learning
=========================================================
Stratégies d'ensemble avancées pour maximiser la performance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import from our main script
exec(open('generate_j6_natural_enhanced.py').read().replace('if __name__ == "__main__":', 'if False:'))

class StackingEnsemble:
    """Stacking ensemble avec méta-learner"""
    
    def __init__(self):
        self.base_models = {}
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        self.trained = False
        
    def fit(self, enhanced_model, augmented_model, original_model, X_meta, y_meta):
        """Entraîner le méta-modèle sur les prédictions des modèles de base"""
        print("🎯 Entraînement Stacking Ensemble...")
        
        self.base_models = {
            'enhanced': enhanced_model,
            'augmented': augmented_model, 
            'original': original_model
        }
        
        # Générer prédictions pour méta-modèle
        meta_features = []
        
        for i in range(len(X_meta)):
            sample_features = []
            
            # Enhanced predictions
            if enhanced_model is not None:
                X_enh = X_meta[i:i+1, :len(enhanced_model.all_features)]
                enh_proba = enhanced_model.predict_proba(X_enh)[0]
                sample_features.extend(enh_proba)
            
            # Augmented predictions  
            if augmented_model is not None:
                # Assume we have the right features for augmented
                aug_proba = augmented_model.predict_proba(X_meta[i:i+1])[0]
                sample_features.extend(aug_proba)
            
            # Original predictions
            if original_model is not None:
                orig_proba = original_model.predict_proba(X_meta[i:i+1])[0]
                sample_features.extend(orig_proba)
            
            meta_features.append(sample_features)
        
        X_meta_features = np.array(meta_features)
        self.meta_model.fit(X_meta_features, y_meta)
        self.trained = True
        
        print(f"✅ Méta-modèle entraîné sur {len(X_meta_features)} échantillons")
        
    def predict(self, X):
        """Prédiction via stacking"""
        if not self.trained:
            raise ValueError("Modèle non entraîné")
        
        meta_features = []
        
        for i in range(len(X)):
            sample_features = []
            
            # Collecter prédictions de tous les modèles de base
            for model_name, model in self.base_models.items():
                if model is not None:
                    proba = model.predict_proba(X[i:i+1])[0]
                    sample_features.extend(proba)
            
            meta_features.append(sample_features)
        
        X_meta = np.array(meta_features)
        return self.meta_model.predict(X_meta)
    
    def predict_proba(self, X):
        """Probabilités via stacking"""
        if not self.trained:
            raise ValueError("Modèle non entraîné")
        
        meta_features = []
        
        for i in range(len(X)):
            sample_features = []
            
            for model_name, model in self.base_models.items():
                if model is not None:
                    proba = model.predict_proba(X[i:i+1])[0]
                    sample_features.extend(proba)
            
            meta_features.append(sample_features)
        
        X_meta = np.array(meta_features)
        return self.meta_model.predict_proba(X_meta)

class AdaptiveWeightedEnsemble:
    """Ensemble avec poids adaptatifs basés sur la confiance"""
    
    def __init__(self):
        self.models = {}
        self.confidence_weights = {}
        
    def fit(self, models_dict):
        """Configurer les modèles"""
        self.models = models_dict
        
        # Poids initiaux basés sur la performance historique
        self.confidence_weights = {
            'enhanced': 0.45,     # Meilleur sur draws
            'augmented': 0.30,    # Bon équilibre
            'original': 0.25      # Stabilité
        }
        
    def predict_with_adaptive_weights(self, X, features_dict):
        """Prédiction avec poids adaptatifs"""
        predictions = {}
        confidences = {}
        
        # Obtenir prédictions de tous les modèles
        for model_name, model in self.models.items():
            if model is not None:
                try:
                    if model_name == 'enhanced':
                        proba = model.predict_proba(X)[0]
                    else:
                        proba = model.predict_proba(X)[0]
                    
                    pred = np.argmax(proba)
                    conf = proba[pred]
                    
                    predictions[model_name] = pred
                    confidences[model_name] = conf
                    
                except Exception as e:
                    predictions[model_name] = 0
                    confidences[model_name] = 0.33
        
        # Adaptation des poids basée sur:
        # 1. Entropie de marché
        # 2. Confiance des modèles
        market_entropy = features_dict.get('market_entropy_historical', 0.5)
        
        adapted_weights = {}
        
        for model_name in self.models:
            base_weight = self.confidence_weights.get(model_name, 0.33)
            conf_bonus = confidences.get(model_name, 0.33) - 0.33
            
            # Boost Enhanced sur haute entropie
            if model_name == 'enhanced' and market_entropy > 0.9:
                entropy_bonus = 0.2
            else:
                entropy_bonus = 0
            
            adapted_weights[model_name] = base_weight + conf_bonus * 0.5 + entropy_bonus
        
        # Normaliser les poids
        total_weight = sum(adapted_weights.values())
        if total_weight > 0:
            for model_name in adapted_weights:
                adapted_weights[model_name] /= total_weight
        
        # Vote pondéré
        final_proba = np.zeros(3)
        
        for model_name, weight in adapted_weights.items():
            if model_name in predictions:
                pred = predictions[model_name]
                # Distribuer le poids selon la confiance
                model_proba = np.zeros(3)
                model_proba[pred] = confidences.get(model_name, 0.33)
                # Distribuer le reste uniformément
                remaining = 1 - model_proba[pred]
                for i in range(3):
                    if i != pred:
                        model_proba[i] = remaining / 2
                
                final_proba += weight * model_proba
        
        final_pred = np.argmax(final_proba)
        
        return {
            'prediction': final_pred,
            'probabilities': final_proba,
            'weights_used': adapted_weights,
            'individual_predictions': predictions,
            'individual_confidences': confidences
        }

def test_advanced_ensembles():
    """Test des stratégies d'ensemble avancées"""
    print("🚀 TEST STRATÉGIES D'ENSEMBLE AVANCÉES")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    epl_2025 = data[data['Season'] == '2025-2026'].copy()
    epl_with_results = epl_2025[epl_2025['FullTimeResult'].notna()]
    real_odds_data = load_real_odds_data()
    
    print(f"✅ Test sur {len(epl_with_results)} matchs EPL 2025-26")
    
    # Create base models
    print("\n🔧 Création des modèles de base...")
    enhanced_cascade, enhanced_features = create_natural_enhanced_cascade()
    augmented_baseline, augmented_features = create_augmented_baseline_champion()
    
    # Train augmented baseline
    historical_data = data[data['Season'] != '2025-2026'].copy()
    X_aug_train = []
    y_aug_train = []
    
    for idx, match in historical_data.head(500).iterrows():  # Sample pour vitesse
        if pd.isna(match['FullTimeResult']):
            continue
            
        try:
            features = calculate_enhanced_features(
                data, match['HomeTeam'], match['AwayTeam'], match['Date'], 
                j6_odds=None, real_odds_data=real_odds_data
            )
            
            feature_vector = []
            for feat_name in augmented_features:
                if feat_name in features:
                    feature_vector.append(features[feat_name])
                elif feat_name == 'market_entropy_norm' and 'market_entropy_historical' in features:
                    feature_vector.append(features['market_entropy_historical'])
                else:
                    feature_vector.append(0.5)
            
            X_aug_train.append(feature_vector)
            result_map = {'H': 0, 'D': 1, 'A': 2}
            y_aug_train.append(result_map[match['FullTimeResult']])
            
        except Exception as e:
            continue
    
    X_aug = np.array(X_aug_train)
    y_aug = np.array(y_aug_train)
    augmented_baseline.fit(X_aug, y_aug)
    
    # Load original baseline
    try:
        original_baseline = joblib.load('models/production/baseline_champion_v23.joblib')
        original_available = True
    except:
        original_baseline = None
        original_available = False
    
    print("✅ Modèles de base créés")
    
    # Create advanced ensembles
    adaptive_ensemble = AdaptiveWeightedEnsemble()
    adaptive_ensemble.fit({
        'enhanced': enhanced_cascade,
        'augmented': augmented_baseline,
        'original': original_baseline
    })
    
    print("✅ Ensemble adaptatif configuré")
    
    # Test ensembles
    print(f"\n🧪 Test des ensembles avancés...")
    
    results = {
        'Enhanced': {'correct': 0, 'total': 0},
        'Adaptive_Ensemble': {'correct': 0, 'total': 0},
        'Weighted_Original': {'correct': 0, 'total': 0}
    }
    
    baseline_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    for idx, match in epl_with_results.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        actual = match['FullTimeResult']
        match_date = match['Date']
        
        try:
            features = calculate_enhanced_features(
                data, home_team, away_team, match_date, 
                j6_odds=None, real_odds_data=real_odds_data
            )
            
            # Enhanced model
            X_enhanced = pd.DataFrame([features])[enhanced_features]
            enhanced_pred = enhanced_cascade.predict(X_enhanced)[0]
            enhanced_class = ['H', 'D', 'A'][enhanced_pred]
            enhanced_correct = (enhanced_class == actual)
            results['Enhanced']['correct'] += enhanced_correct
            results['Enhanced']['total'] += 1
            
            # Adaptive ensemble
            adaptive_result = adaptive_ensemble.predict_with_adaptive_weights(
                X_enhanced, features
            )
            adaptive_class = ['H', 'D', 'A'][adaptive_result['prediction']]
            adaptive_correct = (adaptive_class == actual)
            results['Adaptive_Ensemble']['correct'] += adaptive_correct
            results['Adaptive_Ensemble']['total'] += 1
            
            # Weighted original (baseline de comparaison)
            ensemble = create_weighted_ensemble_model()
            ensemble.fit_models(enhanced_cascade, augmented_baseline, 'models/production/baseline_champion_v23.joblib')
            
            weighted_result = ensemble.predict_with_logic(
                features, enhanced_features, augmented_features, baseline_features
            )
            weighted_class = ['H', 'D', 'A'][weighted_result['prediction']]
            weighted_correct = (weighted_class == actual)
            results['Weighted_Original']['correct'] += weighted_correct
            results['Weighted_Original']['total'] += 1
            
            if results['Enhanced']['total'] % 10 == 0:
                print(f"   Traité {results['Enhanced']['total']} matchs...")
                
        except Exception as e:
            continue
    
    # Results
    print(f"\n🎯 RÉSULTATS ENSEMBLES AVANCÉS:")
    print("=" * 50)
    
    accuracies = {}
    for model_name in results:
        if results[model_name]['total'] > 0:
            accuracy = results[model_name]['correct'] / results[model_name]['total']
            accuracies[model_name] = accuracy
            print(f"{model_name:20}: {results[model_name]['correct']:2}/{results[model_name]['total']:2} = {accuracy:.1%}")
    
    if accuracies:
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model]
        
        print(f"\n🏆 MEILLEUR ENSEMBLE: {best_model} ({best_accuracy:.1%})")
        
        enhanced_baseline = accuracies.get('Enhanced', 0)
        print(f"\n📈 AMÉLIORATIONS vs Enhanced:")
        for model, acc in accuracies.items():
            if model != 'Enhanced':
                improvement = acc - enhanced_baseline
                print(f"   {model}: {improvement:+.1%}")
        
        if best_accuracy >= 0.45:
            print("\n🎉 OBJECTIF 45% ATTEINT avec ensemble avancé!")
        elif best_accuracy > enhanced_baseline:
            print("\n✅ AMÉLIORATION RÉUSSIE avec ensemble!")
    
    return accuracies

if __name__ == "__main__":
    test_advanced_ensembles()