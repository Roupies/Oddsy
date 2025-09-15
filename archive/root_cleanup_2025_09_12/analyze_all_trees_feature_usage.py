#!/usr/bin/env python3
"""
Analyse de l'utilisation des features sur les 300 arbres du Random Forest v2.3
Objectif: Identifier les 3 features les plus utilisées globalement
"""

import joblib
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_all_trees_feature_usage():
    """Analyse l'usage des features sur tous les 300 arbres."""
    
    print("🌳 ANALYSE GLOBALE - USAGE DES FEATURES SUR 300 ARBRES")
    print("="*70)
    
    # Chargement du modèle v2.3 réentraîné
    print("🔧 Chargement du modèle...")
    model = joblib.load('/Users/maxime/Desktop/Oddsy/models/v23_retrained_2025_09_11_154613.joblib')
    rf_model = model.calibrated_classifiers_[0].estimator
    
    print(f"✅ Modèle chargé: {len(rf_model.estimators_)} arbres")
    
    # Features v2.3
    feature_names = [
        "form_diff_normalized", "elo_diff_normalized", "h2h_score", 
        "matchday_normalized", "shots_diff_normalized", "corners_diff_normalized", 
        "market_entropy_norm", "home_xg_eff_10", "away_goals_sum_5", "away_xg_eff_10"
    ]
    
    print(f"📊 Analyse de {len(feature_names)} features sur {len(rf_model.estimators_)} arbres")
    
    # Comptage global des utilisations
    global_feature_usage = defaultdict(int)
    tree_stats = []
    
    print(f"\n🔍 Analyse arbre par arbre...")
    for tree_idx, estimator in enumerate(rf_model.estimators_):
        tree = estimator.tree_
        
        # Comptage pour cet arbre
        tree_feature_count = defaultdict(int)
        
        for node_idx in range(tree.node_count):
            feature_idx = tree.feature[node_idx]
            
            # Si c'est un nœud de décision (pas une feuille)
            if feature_idx >= 0:
                feature_name = feature_names[feature_idx]
                tree_feature_count[feature_name] += 1
                global_feature_usage[feature_name] += 1
        
        # Stats de cet arbre
        tree_stats.append({
            'tree_id': tree_idx + 1,
            'total_splits': sum(tree_feature_count.values()),
            'features_used': len(tree_feature_count),
            'usage': dict(tree_feature_count)
        })
        
        if (tree_idx + 1) % 50 == 0:
            print(f"   📈 Traité {tree_idx + 1}/300 arbres...")
    
    print(f"✅ Analyse terminée!")
    
    # Résultats globaux
    print(f"\n🏆 CLASSEMENT GLOBAL - USAGE SUR 300 ARBRES")
    print("="*60)
    
    # Tri par usage total
    sorted_features = sorted(global_feature_usage.items(), key=lambda x: x[1], reverse=True)
    
    total_splits = sum(global_feature_usage.values())
    
    print(f"📊 Total des splits analysés: {total_splits:,}")
    print(f"🎯 Moyenne par feature: {total_splits/len(feature_names):,.0f}")
    
    print(f"\n🥇🥈🥉 TOP 10 FEATURES (sur 300 arbres):")
    print(f"{'Rang':<4} {'Feature':<25} {'Utilisations':<12} {'%':<8} {'Barres'}")
    print("-" * 70)
    
    for rank, (feature, count) in enumerate(sorted_features, 1):
        percentage = (count / total_splits) * 100
        bar_length = int((count / sorted_features[0][1]) * 30)
        bars = "█" * bar_length
        
        # Marquage des 3 premiers
        if rank <= 3:
            medal = ["🥇", "🥈", "🥉"][rank-1]
        else:
            medal = f"{rank:2}"
        
        print(f"{medal} {feature:<25} {count:>10,} {percentage:>6.1f}% {bars}")
    
    # Focus sur le TOP 3
    print(f"\n🎯 FOCUS SUR LE PODIUM:")
    print("="*50)
    
    top_3 = sorted_features[:3]
    for i, (feature, count) in enumerate(top_3):
        rank_names = ["🥇 CHAMPION", "🥈 VICE-CHAMPION", "🥉 3ème PLACE"]
        percentage = (count / total_splits) * 100
        
        print(f"\n{rank_names[i]}: {feature}")
        print(f"   📊 Utilisations: {count:,} ({percentage:.1f}% des splits)")
        print(f"   🌳 Moyenne par arbre: {count/300:.1f} utilisations")
        
        # Analyse de distribution
        tree_usage = [stats['usage'].get(feature, 0) for stats in tree_stats]
        used_in_trees = sum(1 for usage in tree_usage if usage > 0)
        max_usage = max(tree_usage)
        avg_usage = np.mean([usage for usage in tree_usage if usage > 0])
        
        print(f"   🎯 Présente dans: {used_in_trees}/300 arbres ({used_in_trees/300*100:.1f}%)")
        print(f"   📈 Usage max dans un arbre: {max_usage}")
        print(f"   📊 Usage moyen (quand utilisée): {avg_usage:.1f}")
    
    # Comparaison avec Feature Importance
    print(f"\n🔬 COMPARAISON USAGE vs IMPORTANCE GLOBALE:")
    print("="*60)
    
    # Feature importance du modèle
    global_importance = rf_model.feature_importances_
    importance_ranking = sorted(enumerate(global_importance), key=lambda x: x[1], reverse=True)
    
    print(f"{'Feature':<25} {'Usage Rank':<11} {'Importance Rank':<15} {'Différence'}")
    print("-" * 70)
    
    for feature_idx, feature_name in enumerate(feature_names):
        # Rang dans usage
        usage_rank = next(i for i, (fname, _) in enumerate(sorted_features, 1) if fname == feature_name)
        
        # Rang dans importance
        importance_rank = next(i for i, (idx, _) in enumerate(importance_ranking, 1) if idx == feature_idx)
        
        diff = abs(usage_rank - importance_rank)
        diff_str = f"+{diff}" if usage_rank > importance_rank else f"-{diff}" if usage_rank < importance_rank else "="
        
        print(f"{feature_name:<25} #{usage_rank:<10} #{importance_rank:<14} {diff_str}")
    
    # Graphique de visualisation
    create_usage_visualization(sorted_features, feature_names)
    
    return sorted_features, tree_stats

def create_usage_visualization(sorted_features, all_features):
    """Crée une visualisation de l'usage des features."""
    
    print(f"\n📊 Génération de la visualisation...")
    
    plt.figure(figsize=(14, 8))
    
    features = [f[0] for f in sorted_features]
    counts = [f[1] for f in sorted_features]
    
    # Couleurs spéciales pour le podium
    colors = []
    for i, _ in enumerate(features):
        if i == 0:
            colors.append('#FFD700')  # Or
        elif i == 1:
            colors.append('#C0C0C0')  # Argent
        elif i == 2:
            colors.append('#CD7F32')  # Bronze
        else:
            colors.append('#87CEEB')  # Bleu clair
    
    bars = plt.barh(features, counts, color=colors, edgecolor='navy', linewidth=0.8)
    
    # Annotations sur les barres
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        plt.text(width + 100, bar.get_y() + bar.get_height()/2, 
                f'{count:,}', ha='left', va='center', fontweight='bold')
    
    plt.xlabel('Nombre Total d\'Utilisations (300 arbres)', fontsize=12, fontweight='bold')
    plt.title('Usage des Features - Random Forest v2.3 (300 arbres)\n🥇🥈🥉 Podium des Champions', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Ligne de moyenne
    avg_usage = np.mean(counts)
    plt.axvline(avg_usage, color='red', linestyle='--', alpha=0.7, 
                label=f'Moyenne: {avg_usage:,.0f}')
    
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = '/Users/maxime/Desktop/Oddsy/global_feature_usage_300_trees.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Graphique sauvegardé: {output_path}")

def main():
    """Fonction principale."""
    
    print("🎯 MISSION: Identifier les 3 features championnes sur 300 arbres")
    print("🔍 Méthode: Comptage exhaustif des splits dans chaque arbre")
    print("="*70)
    
    # Analyse complète
    sorted_features, tree_stats = analyze_all_trees_feature_usage()
    
    # Résumé final
    print(f"\n🏆 PODIUM FINAL - TOP 3 FEATURES:")
    print("="*50)
    
    top_3 = sorted_features[:3]
    medals = ["🥇", "🥈", "🥉"]
    
    for i, (feature, count) in enumerate(top_3):
        print(f"{medals[i]} {i+1}. {feature}: {count:,} utilisations")
    
    total_top3 = sum(count for _, count in top_3)
    total_all = sum(count for _, count in sorted_features)
    
    print(f"\n📊 Le podium représente {total_top3/total_all*100:.1f}% de tous les splits!")
    print(f"✅ Analyse terminée - Les champions sont identifiés!")

if __name__ == "__main__":
    main()