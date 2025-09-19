#!/usr/bin/env python3
"""
🌳 Visualisation d'un arbre de décision du modèle Baseline Champion v2.3
Génère une prévisualisation interactive d'un arbre représentatif
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt
from dtreeviz import dtreeviz
import os

def load_production_data():
    """Charge les données de production pour la visualisation."""
    print("📊 Chargement des données de production...")
    
    # Chercher le dataset le plus récent
    data_files = [
        'data/processed/v_auto_update_20250916_110247.csv',
        'data/processed/v16_contextual_metadata_20250915_171540.json',
        'data/processed/v15_final_enhanced.csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ Utilisation du dataset: {file_path}")
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            else:
                return pd.read_json(file_path)
    
    print("⚠️ Aucun dataset trouvé, génération de données exemple...")
    return None

def visualize_champion_tree():
    """Visualise un arbre du modèle champion avec dtreeviz."""
    
    print("🏆 VISUALISATION ARBRE - BASELINE CHAMPION v2.3")
    print("="*60)
    
    # Chargement du modèle champion
    model_path = 'models/production/baseline_champion_v23.joblib'
    if not os.path.exists(model_path):
        # Fallback vers le modèle standard
        model_path = 'models/v23_retrained_2025_09_11_154613.joblib'
    
    print(f"📁 Chargement du modèle: {model_path}")
    model = joblib.load(model_path)
    
    # Extraction du RandomForest depuis le CalibratedClassifier
    if hasattr(model, 'calibrated_classifiers_'):
        rf_model = model.calibrated_classifiers_[0].estimator
        print("✅ Modèle RandomForest extrait du CalibratedClassifier")
    else:
        rf_model = model
        print("✅ Modèle RandomForest direct")
    
    # Features de production (les 10 validées)
    feature_names = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    target_names = ['Home', 'Draw', 'Away']
    
    # Sélection d'un arbre représentatif (milieu de la forêt)
    tree_idx = 50  # Arbre 50 sur 300
    selected_tree = rf_model.estimators_[tree_idx]
    
    print(f"🌳 Arbre sélectionné: #{tree_idx}")
    print(f"📏 Profondeur max: {selected_tree.tree_.max_depth}")
    print(f"🍃 Nombre de feuilles: {selected_tree.tree_.n_leaves}")
    print(f"📊 Nombre de nœuds: {selected_tree.tree_.node_count}")
    
    # Chargement des données pour la visualisation
    data = load_production_data()
    
    if data is not None and len(feature_names) <= len(data.columns):
        try:
            # Préparation des données pour dtreeviz
            X = data[feature_names].fillna(0)
            y = data['target'] if 'target' in data.columns else np.random.randint(0, 3, len(X))
            
            print("🎨 Génération de la visualisation dtreeviz...")
            
            # Création de la visualisation interactive
            viz = dtreeviz(
                selected_tree,
                X.values,
                y,
                target_name="Match_Result",
                feature_names=feature_names,
                class_names=target_names,
                title=f"🏆 Baseline Champion v2.3 - Arbre #{tree_idx}",
                fancy=True,
                scale=1.2,
                label_fontsize=10,
                orientation="TD"  # Top-Down
            )
            
            # Sauvegarde
            output_file = "tree_visualization_champion.svg"
            viz.save(output_file)
            print(f"💾 Visualisation sauvée: {output_file}")
            
        except Exception as e:
            print(f"⚠️ Erreur dtreeviz: {e}")
            print("🔄 Fallback vers matplotlib...")
    
    # Visualisation alternative avec matplotlib
    print("🎨 Génération de la visualisation matplotlib...")
    
    plt.figure(figsize=(20, 12))
    plot_tree(
        selected_tree,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=4  # Limitation pour la lisibilité
    )
    
    plt.title(f"🏆 Baseline Champion v2.3 - Arbre #{tree_idx} (Profondeur limitée à 4)", 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarde
    output_file = "tree_visualization_matplotlib.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Visualisation matplotlib sauvée: {output_file}")
    
    # Affichage des règles textuelles pour les premiers niveaux
    print("\n📋 RÈGLES DE DÉCISION (4 premiers niveaux):")
    print("-" * 50)
    
    tree_rules = export_text(
        selected_tree,
        feature_names=feature_names,
        max_depth=4,
        spacing=3,
        decimals=3,
        show_weights=True
    )
    
    print(tree_rules)
    
    # Analyse des features importantes dans cet arbre
    print("\n🔍 ANALYSE DES FEATURES DANS CET ARBRE:")
    print("-" * 50)
    
    tree_feature_importance = selected_tree.tree_.compute_feature_importances(normalize=False)
    
    for i, importance in enumerate(tree_feature_importance):
        if importance > 0:
            print(f"📊 {feature_names[i]}: {importance:.4f}")
    
    print(f"\n✅ Visualisation terminée!")
    print(f"📁 Fichiers générés:")
    print(f"   - tree_visualization_matplotlib.png")
    if os.path.exists("tree_visualization_champion.svg"):
        print(f"   - tree_visualization_champion.svg")

if __name__ == "__main__":
    visualize_champion_tree()