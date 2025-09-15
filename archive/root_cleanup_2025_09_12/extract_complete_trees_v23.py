#!/usr/bin/env python3
"""
Script avancé pour extraire des arbres complets (profondeur 20) du modèle v2.3
Génère des diagrammes ASCII détaillés et des données de calibration
"""

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import json

def extract_complete_tree_structure(tree, feature_names, max_depth=None, detailed=False):
    """Extrait la structure complète d'un arbre avec tous les détails."""
    
    def get_detailed_node_info(node_id, depth=0, prefix="", is_last=True, path_conditions=[]):
        if max_depth is not None and depth > max_depth:
            return "    ... (arbre continue plus profond)\n"
            
        # Informations du noeud
        feature_id = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        samples = tree.n_node_samples[node_id]
        values = tree.value[node_id][0]
        
        # Calcul des probabilités et statistiques
        total_samples = np.sum(values)
        probabilities = values / total_samples if total_samples > 0 else [0, 0, 0]
        
        # Calcul de l'entropie (mesure d'impureté)
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Symboles pour l'arbre ASCII
        connector = "└── " if is_last else "├── "
        
        result = ""
        
        # Si c'est une feuille (pas de split)
        if feature_id < 0:
            prediction = np.argmax(values)
            prediction_names = ["HOME", "DRAW", "AWAY"]
            confidence = probabilities[prediction]
            
            result += f"{prefix}{connector}🍃 **FEUILLE FINALE**: {prediction_names[prediction]}\n"
            result += f"{prefix}{'    ' if is_last else '│   '}   📊 Confiance: {confidence:.1%} | Échantillons: {samples}\n"
            result += f"{prefix}{'    ' if is_last else '│   '}   📈 Probabilités: HOME={probabilities[0]:.1%}, DRAW={probabilities[1]:.1%}, AWAY={probabilities[2]:.1%}\n"
            result += f"{prefix}{'    ' if is_last else '│   '}   🎯 Entropie: {entropy:.3f} (pureté: {1-entropy/1.585:.1%})\n"
            
            if detailed and path_conditions:
                result += f"{prefix}{'    ' if is_last else '│   '}   🔍 Chemin suivi:\n"
                for i, condition in enumerate(path_conditions):
                    result += f"{prefix}{'    ' if is_last else '│   '}      {i+1}. {condition}\n"
            
            return result
        
        # Noeud de décision
        feature_name = feature_names[feature_id]
        result += f"{prefix}{connector}❓ **{feature_name}** <= {threshold:.3f} ?\n"
        result += f"{prefix}{'    ' if is_last else '│   '}   📊 Échantillons: {samples} | Entropie: {entropy:.3f}\n"
        result += f"{prefix}{'    ' if is_last else '│   '}   📈 Distribution: HOME={probabilities[0]:.1%}, DRAW={probabilities[1]:.1%}, AWAY={probabilities[2]:.1%}\n"
        
        # Gini impurity pour comparaison
        gini = 1 - sum(p**2 for p in probabilities)
        result += f"{prefix}{'    ' if is_last else '│   '}   🎲 Gini: {gini:.3f} | Gain potentiel: {entropy - gini:.3f}\n"
        
        # Branches enfants
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        # Conditions pour le chemin
        left_condition = f"{feature_name} <= {threshold:.3f}"
        right_condition = f"{feature_name} > {threshold:.3f}"
        
        # Branche OUI (left)
        result += f"{new_prefix}├── ✅ OUI: {left_condition}\n"
        left_path = path_conditions + [left_condition] if detailed else []
        result += get_detailed_node_info(left_child, depth + 1, new_prefix + "│   ", False, left_path)
        
        # Branche NON (right)  
        result += f"{new_prefix}└── ❌ NON: {right_condition}\n"
        right_path = path_conditions + [right_condition] if detailed else []
        result += get_detailed_node_info(right_child, depth + 1, new_prefix + "    ", True, right_path)
        
        return result
    
    return get_detailed_node_info(0, path_conditions=[] if detailed else [])

def generate_calibration_examples():
    """Génère des exemples détaillés de calibration."""
    
    print("🎯 Génération d'exemples de calibration...")
    
    # Chargement du modèle
    model = joblib.load('/Users/maxime/Desktop/Oddsy/models/randomforest_corrected_model_2025_09_02_113228.joblib')
    rf_model = model.calibrated_classifiers_[0].estimator
    calibrator = model.calibrated_classifiers_[0].calibrators[0]
    
    # Noms des features
    feature_names = [
        "form_diff_normalized", "elo_diff_normalized", "h2h_score", "matchday_normalized",
        "shots_diff_normalized", "corners_diff_normalized", "market_entropy_norm",
        "home_xg_eff_10", "away_goals_sum_5", "away_xg_eff_10"
    ]
    
    # Exemples de matchs avec différents niveaux de confiance
    examples = [
        {
            "name": "Man City (dom) vs Burnley (ext) - Favori Clair",
            "features": [0.75, 0.85, 0.65, 0.40, 0.78, 0.72, 0.15, 1.25, 4.0, 0.85]
        },
        {
            "name": "Liverpool (dom) vs Arsenal (ext) - Match Équilibré", 
            "features": [0.55, 0.58, 0.48, 0.60, 0.52, 0.55, 0.65, 1.05, 7.0, 1.15]
        },
        {
            "name": "Brentford (dom) vs Man United (ext) - Outsider",
            "features": [0.25, 0.35, 0.35, 0.75, 0.38, 0.42, 0.85, 0.88, 5.0, 1.08]
        },
        {
            "name": "Brighton (dom) vs Wolves (ext) - Incertain",
            "features": [0.48, 0.52, 0.50, 0.30, 0.45, 0.48, 0.95, 0.95, 6.0, 0.98]
        }
    ]
    
    calibration_data = []
    
    for example in examples:
        features_array = np.array([example["features"]])
        
        # Probabilités brutes de la Random Forest
        raw_probabilities = rf_model.predict_proba(features_array)[0]
        
        # Probabilités calibrées
        calibrated_probabilities = model.predict_proba(features_array)[0]
        
        calibration_data.append({
            "match": example["name"],
            "features": dict(zip(feature_names, example["features"])),
            "raw_probabilities": {
                "HOME": float(raw_probabilities[0]),
                "DRAW": float(raw_probabilities[1]), 
                "AWAY": float(raw_probabilities[2])
            },
            "calibrated_probabilities": {
                "HOME": float(calibrated_probabilities[0]),
                "DRAW": float(calibrated_probabilities[1]),
                "AWAY": float(calibrated_probabilities[2])
            },
            "calibration_effect": {
                "HOME": float(calibrated_probabilities[0] - raw_probabilities[0]),
                "DRAW": float(calibrated_probabilities[1] - raw_probabilities[1]),
                "AWAY": float(calibrated_probabilities[2] - raw_probabilities[2])
            }
        })
    
    return calibration_data

def analyze_complete_trees():
    """Analyse complète avec arbres détaillés."""
    
    print("🌳 Extraction d'arbres complets (profondeur 20)...")
    
    # Chargement du modèle
    model = joblib.load('/Users/maxime/Desktop/Oddsy/models/randomforest_corrected_model_2025_09_02_113228.joblib')
    rf_model = model.calibrated_classifiers_[0].estimator
    
    # Noms des features
    feature_names = [
        "form_diff_normalized", "elo_diff_normalized", "h2h_score", "matchday_normalized",
        "shots_diff_normalized", "corners_diff_normalized", "market_entropy_norm",
        "home_xg_eff_10", "away_goals_sum_5", "away_xg_eff_10"
    ]
    
    print(f"✅ Modèle chargé: {len(rf_model.estimators_)} arbres")
    
    # Sélection d'arbres avec différentes caractéristiques
    selected_trees = [
        {"idx": 0, "name": "Arbre #1 - Focus Elo/Force", "max_depth": 8},
        {"idx": 25, "name": "Arbre #26 - Approche Équilibrée", "max_depth": 10},
        {"idx": 50, "name": "Arbre #51 - Spécialiste Forme", "max_depth": 12},
        {"idx": 100, "name": "Arbre #101 - Analyseur Marché", "max_depth": 15},
        {"idx": 150, "name": "Arbre #151 - Expert xG", "max_depth": None},  # Arbre complet!
        {"idx": 200, "name": "Arbre #201 - Généraliste", "max_depth": 10},
        {"idx": 299, "name": "Arbre #300 - Dernier Spécimen", "max_depth": 8}
    ]
    
    results = {
        "model_info": {
            "n_estimators": len(rf_model.estimators_),
            "max_depth": rf_model.max_depth,
            "max_features": rf_model.max_features,
            "min_samples_split": rf_model.min_samples_split,
            "class_weight": str(rf_model.class_weight)
        },
        "feature_names": feature_names,
        "complete_trees": [],
        "calibration_examples": generate_calibration_examples()
    }
    
    for tree_info in selected_trees:
        idx = tree_info["idx"]
        tree = rf_model.estimators_[idx].tree_
        
        print(f"\n{'='*60}")
        print(f"🌲 {tree_info['name']}")
        print(f"📊 Nœuds: {tree.node_count} | Profondeur max: {tree.max_depth}")
        print(f"🎯 Profondeur affichée: {'COMPLÈTE' if tree_info['max_depth'] is None else tree_info['max_depth']}")
        print(f"{'='*60}")
        
        # Extraction avec niveau de détail approprié
        detailed = (idx == 150)  # Arbre #151 avec tous les détails
        structure = extract_complete_tree_structure(
            tree, feature_names, 
            max_depth=tree_info["max_depth"],
            detailed=detailed
        )
        
        print(structure)
        
        results["complete_trees"].append({
            "tree_number": idx + 1,
            "name": tree_info["name"],
            "structure": structure,
            "total_nodes": tree.node_count,
            "max_depth_reached": tree.max_depth,
            "display_depth": tree_info["max_depth"] or tree.max_depth,
            "is_complete": tree_info["max_depth"] is None
        })
    
    # Sauvegarde des résultats
    with open('/Users/maxime/Desktop/Oddsy/complete_trees_analysis_v23.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Analyse complète terminée!")
    print(f"📁 Résultats sauvés dans 'complete_trees_analysis_v23.json'")
    print(f"🌳 {len(selected_trees)} arbres analysés")
    print(f"📊 {len(results['calibration_examples'])} exemples de calibration générés")
    
    return results

if __name__ == "__main__":
    analyze_complete_trees()