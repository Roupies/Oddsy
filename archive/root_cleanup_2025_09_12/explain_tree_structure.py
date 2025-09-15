#!/usr/bin/env python3
"""
Explication détaillée de la structure interne des arbres de décision
Focus: Nœuds, feuilles, symétrie et logique de navigation
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_tree_structure():
    """Analyse la structure interne des arbres pour expliquer nœuds/feuilles."""
    
    print("🌳 STRUCTURE INTERNE DES ARBRES - EXPLICATION DÉTAILLÉE")
    print("="*70)
    
    # Chargement du modèle
    model = joblib.load('/Users/maxime/Desktop/Oddsy/models/v23_retrained_2025_09_11_154613.joblib')
    rf_model = model.calibrated_classifiers_[0].estimator
    
    # Analyse de plusieurs arbres pour montrer la variété
    trees_to_analyze = [0, 25, 50, 150, 299]  # 5 arbres différents
    
    print("📊 ANALYSE DE 5 ARBRES REPRÉSENTATIFS")
    print("-" * 50)
    
    for tree_idx in trees_to_analyze:
        tree = rf_model.estimators_[tree_idx].tree_
        
        # Comptage des nœuds
        total_nodes = tree.node_count
        internal_nodes = 0
        leaf_nodes = 0
        
        for i in range(total_nodes):
            if tree.children_left[i] == -1:  # C'est une feuille
                leaf_nodes += 1
            else:
                internal_nodes += 1
        
        # Analyse de la profondeur
        max_depth = tree.max_depth
        
        # Calcul de symétrie (théorique vs réelle)
        theoretical_nodes = (2 ** (max_depth + 1)) - 1
        symmetry_ratio = total_nodes / theoretical_nodes
        
        print(f"🌲 Arbre #{tree_idx + 1}:")
        print(f"   📊 Nœuds totaux: {total_nodes}")
        print(f"   🔧 Nœuds internes (décision): {internal_nodes}")
        print(f"   🍃 Feuilles (prédiction): {leaf_nodes}")
        print(f"   📏 Profondeur max: {max_depth}")
        print(f"   ⚖️ Symétrie: {symmetry_ratio:.1%} (1.0 = parfaitement symétrique)")
        
        if tree_idx == 150:  # Analyse détaillée de l'Arbre #151
            analyze_tree_asymmetry(tree, tree_idx + 1)
        
        print()

def analyze_tree_asymmetry(tree, tree_number):
    """Analyse détaillée de l'asymétrie d'un arbre."""
    
    print(f"   🔍 ANALYSE DÉTAILLÉE ASYMÉTRIE - Arbre #{tree_number}")
    
    # Analyse par niveau de profondeur
    nodes_by_depth = defaultdict(int)
    max_possible_by_depth = {}
    
    def traverse_tree(node_id, current_depth):
        nodes_by_depth[current_depth] += 1
        
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        
        if left != -1:  # Pas une feuille
            traverse_tree(left, current_depth + 1)
            traverse_tree(right, current_depth + 1)
    
    traverse_tree(0, 0)  # Commence à la racine
    
    print(f"   📋 Répartition par niveau:")
    for depth in sorted(nodes_by_depth.keys()):
        actual = nodes_by_depth[depth]
        theoretical = 2 ** depth  # Maximum possible à ce niveau
        ratio = actual / theoretical
        print(f"      Niveau {depth}: {actual}/{theoretical} nœuds ({ratio:.1%})")

def explain_tree_logic():
    """Explique la logique de navigation dans un arbre."""
    
    print("\n🧠 LOGIQUE DE NAVIGATION DANS UN ARBRE")
    print("="*60)
    
    print("""
🌳 STRUCTURE D'UN ARBRE DE DÉCISION:

┌─────────────────┐
│    RACINE       │ ← Tous les matchs commencent ici
│ elo_diff ≤ 0.5? │
└─────────────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼───┐
│  OUI  │ │  NON  │ ← DIVISION des données
│       │ │       │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
[Sous-arbres] [Sous-arbres] ← Processus récursif

🔑 CONCEPTS FONDAMENTAUX:

1️⃣ NŒUD INTERNE (Décision):
   • Pose une question: "feature ≤ seuil ?"
   • A exactement 2 enfants: Gauche (OUI) et Droite (NON)
   • Divise les données en 2 groupes

2️⃣ FEUILLE (Prédiction finale):
   • N'a PAS d'enfants (children_left = children_right = -1)
   • Contient la prédiction: HOME/DRAW/AWAY
   • Fin du chemin de décision

3️⃣ CHEMIN DE DÉCISION:
   • 1 seul chemin par prédiction: Racine → ... → Feuille
   • Chaque match suit UN SEUL chemin
   • Pas de "retour en arrière"
    """)

def explain_asymmetry():
    """Explique pourquoi les arbres ne sont pas symétriques."""
    
    print("\n⚖️ POURQUOI LES ARBRES NE SONT-ILS PAS SYMÉTRIQUES ?")
    print("="*65)
    
    print("""
🎯 RAISONS DE L'ASYMÉTRIE:

1️⃣ ARRÊT PRÉCOCE (Early Stopping):
   ✅ Si un nœud devient "pur" (ex: 100% HOME), il devient feuille
   ✅ Pas besoin de diviser plus: économie de calcul
   
   Exemple:
   ┌──────────────┐
   │ elo_diff ≤ 0.2 │ ← 95% des matchs sont AWAY
   └──────┬───────┘
          │
      ┌───▼───┐
      │ FEUILLE │ ← Arrêt ici! Pas besoin de continuer
      │  AWAY   │
      └───────┘

2️⃣ CRITÈRES D'ARRÊT:
   • min_samples_split = 5: Moins de 5 matchs → Arrêt
   • Pureté suffisante (Gini faible) → Arrêt
   • Profondeur maximale atteinte → Arrêt

3️⃣ DONNÉES RÉELLES DÉSÉQUILIBRÉES:
   • HOME: 44%, DRAW: 23%, AWAY: 33%
   • Certaines branches deviennent pures plus vite
   • Asymétrie naturelle reflète la réalité du football

📊 COMPARAISON SYMÉTRIQUE vs RÉELLE:

ARBRE THÉORIQUE (Symétrique):     ARBRE RÉEL (Asymétrique):
      Racine                           Racine
     /      \\                         /      \\
    N1      N2                       N1      FEUILLE_A
   / \\     / \\                     / \\     (Arrêt précoce)
  F1 F2   F3 F4                   N3  F2
                                  /
                                F5
                                
• 7 nœuds total                  • 5 nœuds total
• Toujours équilibré             • Structure adaptée aux données
    """)

def explain_path_logic():
    """Explique pourquoi on suit un seul chemin."""
    
    print("\n🛤️ POURQUOI UN SEUL CHEMIN PAR MATCH ?")
    print("="*50)
    
    print("""
🎯 LOGIQUE DU CHEMIN UNIQUE:

Imaginons un match: Man City (domicile) vs Burnley (extérieur)
Caractéristiques: elo_diff = 0.75, xG_eff = 1.2, entropy = 0.3

🌳 NAVIGATION DANS L'ARBRE:

Étape 1: RACINE
├─ Question: "elo_diff_normalized ≤ 0.5 ?"
├─ Valeur match: 0.75
├─ Réponse: NON (0.75 > 0.5)
└─ ➡️ VA À DROITE (branche NON)

Étape 2: NŒUD DROIT
├─ Question: "home_xg_eff_10 ≤ 1.1 ?"
├─ Valeur match: 1.2  
├─ Réponse: NON (1.2 > 1.1)
└─ ➡️ VA À DROITE encore

Étape 3: FEUILLE
├─ Plus de questions
├─ Prédiction: HOME (87% confiance)
└─ ✅ FIN DU CHEMIN

🔑 POINTS CLÉS:

❌ PAS de "essayer l'autre branche"
   • Un match a des valeurs FIXES
   • 0.75 > 0.5 sera TOUJOURS vrai
   • Pas d'ambiguïté possible

✅ DÉTERMINISME TOTAL:
   • Même match = Même chemin = Même prédiction
   • Reproductibilité garantie
   • Pas de hasard dans la navigation

🌟 POURQUOI C'EST LOGIQUE:
   • Chaque branche représente un PROFIL de match différent
   • Gauche: "Équipes équilibrées" (elo ≤ 0.5)
   • Droite: "Domicile favori" (elo > 0.5)
   • Un match ne peut être que dans UN profil à la fois !
    """)

def create_structure_diagram():
    """Crée un diagramme explicatif de la structure."""
    
    print(f"\n📊 Génération du diagramme explicatif...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Diagramme 1: Structure théorique vs réelle
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.set_title("Arbre Symétrique vs Asymétrique", fontsize=14, fontweight='bold')
    
    # Arbre symétrique (gauche)
    ax1.text(2, 7, "THÉORIQUE", ha='center', fontweight='bold', fontsize=12)
    ax1.text(2, 6, "Racine", ha='center', bbox=dict(boxstyle="round", facecolor='lightblue'))
    
    # Niveau 1
    ax1.text(1, 5, "N1", ha='center', bbox=dict(boxstyle="round", facecolor='lightgreen'))
    ax1.text(3, 5, "N2", ha='center', bbox=dict(boxstyle="round", facecolor='lightgreen'))
    ax1.plot([2, 1], [6, 5], 'k-')
    ax1.plot([2, 3], [6, 5], 'k-')
    
    # Niveau 2
    ax1.text(0.5, 4, "F1", ha='center', bbox=dict(boxstyle="round", facecolor='yellow'))
    ax1.text(1.5, 4, "F2", ha='center', bbox=dict(boxstyle="round", facecolor='yellow'))
    ax1.text(2.5, 4, "F3", ha='center', bbox=dict(boxstyle="round", facecolor='yellow'))
    ax1.text(3.5, 4, "F4", ha='center', bbox=dict(boxstyle="round", facecolor='yellow'))
    
    ax1.plot([1, 0.5], [5, 4], 'k-')
    ax1.plot([1, 1.5], [5, 4], 'k-')
    ax1.plot([3, 2.5], [5, 4], 'k-')
    ax1.plot([3, 3.5], [5, 4], 'k-')
    
    # Arbre asymétrique (droite)
    ax1.text(7, 7, "RÉEL", ha='center', fontweight='bold', fontsize=12)
    ax1.text(7, 6, "Racine", ha='center', bbox=dict(boxstyle="round", facecolor='lightblue'))
    
    ax1.text(6, 5, "N1", ha='center', bbox=dict(boxstyle="round", facecolor='lightgreen'))
    ax1.text(8, 5, "F_PURE", ha='center', bbox=dict(boxstyle="round", facecolor='orange'))
    ax1.plot([7, 6], [6, 5], 'k-')
    ax1.plot([7, 8], [6, 5], 'k-')
    
    ax1.text(5.5, 4, "N2", ha='center', bbox=dict(boxstyle="round", facecolor='lightgreen'))
    ax1.text(6.5, 4, "F2", ha='center', bbox=dict(boxstyle="round", facecolor='yellow'))
    ax1.plot([6, 5.5], [5, 4], 'k-')
    ax1.plot([6, 6.5], [5, 4], 'k-')
    
    ax1.text(5, 3, "F3", ha='center', bbox=dict(boxstyle="round", facecolor='yellow'))
    ax1.text(6, 3, "F4", ha='center', bbox=dict(boxstyle="round", facecolor='yellow'))
    ax1.plot([5.5, 5], [4, 3], 'k-')
    ax1.plot([5.5, 6], [4, 3], 'k-')
    
    ax1.text(2, 2, "7 nœuds\nParfaitement équilibré", ha='center', fontsize=10)
    ax1.text(7, 2, "6 nœuds\nAdapté aux données", ha='center', fontsize=10)
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    
    # Diagramme 2: Chemin unique
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title("Chemin Unique pour un Match", fontsize=14, fontweight='bold')
    
    # Match example
    ax2.text(5, 9.5, "MATCH: City vs Burnley", ha='center', fontweight='bold', 
            bbox=dict(boxstyle="round", facecolor='lightcoral'))
    ax2.text(5, 9, "elo_diff=0.75, xG=1.2", ha='center', fontsize=10)
    
    # Tree navigation
    ax2.text(5, 8, "elo_diff ≤ 0.5 ?", ha='center', bbox=dict(boxstyle="round", facecolor='lightblue'))
    
    ax2.text(2, 6.5, "OUI\n(≤ 0.5)", ha='center', bbox=dict(boxstyle="round", facecolor='lightgray'))
    ax2.text(8, 6.5, "NON\n(> 0.5)", ha='center', bbox=dict(boxstyle="round", facecolor='lightgreen'))
    
    ax2.plot([5, 2], [8, 6.5], 'k--', alpha=0.3, linewidth=2)
    ax2.plot([5, 8], [8, 6.5], 'r-', linewidth=3)
    
    ax2.text(8, 5, "xG ≤ 1.1 ?", ha='center', bbox=dict(boxstyle="round", facecolor='lightblue'))
    ax2.plot([8, 8], [6.5, 5], 'r-', linewidth=3)
    
    ax2.text(6.5, 3.5, "OUI", ha='center', bbox=dict(boxstyle="round", facecolor='lightgray'))
    ax2.text(9.5, 3.5, "NON", ha='center', bbox=dict(boxstyle="round", facecolor='lightgreen'))
    
    ax2.plot([8, 6.5], [5, 3.5], 'k--', alpha=0.3, linewidth=2)
    ax2.plot([8, 9.5], [5, 3.5], 'r-', linewidth=3)
    
    ax2.text(9.5, 2, "FEUILLE\nHOME 87%", ha='center', 
            bbox=dict(boxstyle="round", facecolor='gold'))
    ax2.plot([9.5, 9.5], [3.5, 2], 'r-', linewidth=3)
    
    # Legend
    ax2.text(1, 1, "━ Chemin suivi\n┅ Chemin ignoré", fontsize=10,
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/maxime/Desktop/Oddsy/tree_structure_explanation.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Diagramme sauvegardé: tree_structure_explanation.png")

def main():
    """Fonction principale d'explication."""
    
    print("🎯 MISSION: Expliquer la structure interne des arbres")
    print("🔍 Focus: Nœuds, feuilles, symétrie, chemins de décision")
    print("="*70)
    
    # Analyses détaillées
    analyze_tree_structure()
    explain_tree_logic()
    explain_asymmetry()
    explain_path_logic()
    create_structure_diagram()
    
    print("\n✅ RÉSUMÉ FINAL:")
    print("="*40)
    print("""
🌳 STRUCTURE:
   • Nœuds internes = Questions/Décisions
   • Feuilles = Prédictions finales
   • Asymétrie = Normal et intelligent

🛤️ NAVIGATION:
   • 1 match = 1 seul chemin possible
   • Déterminisme total
   • Pas de "retour en arrière"

⚖️ ASYMÉTRIE:
   • Arrêt précoce quand c'est logique
   • Économie de calcul
   • Adaptation aux vraies données
    """)

if __name__ == "__main__":
    main()