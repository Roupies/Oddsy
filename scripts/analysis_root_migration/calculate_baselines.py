import pandas as pd
import numpy as np

def calculate_baselines(filepath):
    """Calcule les baselines exactes pour évaluer la performance du modèle"""
    print("📊 CALCUL DES BASELINES ODDSY")
    print("=" * 50)
    
    # Load dataset
    df = pd.read_csv(filepath)
    print(f"Dataset: {len(df)} matchs")
    
    # Distribution réelle
    result_counts = df['FullTimeResult'].value_counts()
    total = len(df)
    
    print(f"\n📈 DISTRIBUTION RÉELLE:")
    probs = {}
    for result in ['H', 'D', 'A']:
        count = result_counts.get(result, 0)
        prob = count / total
        probs[result] = prob
        print(f"   {result}: {count:4d} matchs ({prob:.1%})")
    
    # Baseline 1: Hasard pur (33/33/33)
    baseline_random = 1/3
    print(f"\n🎲 BASELINE 1 - HASARD PUR:")
    print(f"   Accuracy attendue: {baseline_random:.1%}")
    
    # Baseline 2: Toujours prédire classe majoritaire
    majority_class = result_counts.idxmax()
    baseline_majority = result_counts.max() / total
    print(f"\n🏆 BASELINE 2 - CLASSE MAJORITAIRE:")
    print(f"   Toujours prédire '{majority_class}': {baseline_majority:.1%}")
    
    # Baseline 3: Prédiction probabiliste selon distribution
    # Si on prédit aléatoirement selon les vraies probabilités
    baseline_prob = sum(prob**2 for prob in probs.values())
    print(f"\n🎯 BASELINE 3 - HASARD PONDÉRÉ:")
    print(f"   Prédiction selon distribution réelle: {baseline_prob:.1%}")
    
    # Baseline 4: Stratégie "intelligente" simple
    # Toujours Home sauf si équipe visiteur très forte (simulation)
    home_wins = result_counts.get('H', 0)
    baseline_home_bias = home_wins / total
    print(f"\n🏠 BASELINE 4 - BIAIS DOMICILE:")
    print(f"   Toujours prédire Home: {baseline_home_bias:.1%}")
    
    # Objectifs pour le modèle ML
    print(f"\n🎯 OBJECTIFS POUR TON MODÈLE ML:")
    print(f"   • Minimum acceptable: > {max(baseline_majority, baseline_prob):.1%}")
    print(f"   • Bon modèle: > 50%")
    print(f"   • Excellent modèle: > 55%")
    
    # F1-Score par classe (pour gérer déséquilibre)
    print(f"\n📏 MÉTRIQUES RECOMMANDÉES:")
    print(f"   • Accuracy globale")
    print(f"   • F1-macro (moyenne F1 de H/D/A)")
    print(f"   • Confusion Matrix détaillée")
    print(f"   • Précision sur classe 'D' (la plus difficile)")
    
    return {
        'random': baseline_random,
        'majority': baseline_majority,
        'weighted': baseline_prob,
        'home_bias': baseline_home_bias,
        'distribution': probs
    }

if __name__ == "__main__":
    # Calculer sur le dataset ML-ready
    baselines = calculate_baselines('/Users/maxime/Desktop/Oddsy/data/processed/premier_league_ml_ready.csv')
    
    print(f"\n💡 RÉSUMÉ:")
    print(f"   Ton modèle DOIT battre: {max(baselines['majority'], baselines['weighted']):.1%}")
    print(f"   Objectif réaliste: ~52-55%")
    print(f"   Si > 55%: très bon modèle pour prédiction foot!")