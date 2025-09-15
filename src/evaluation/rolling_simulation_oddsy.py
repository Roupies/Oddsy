#!/usr/bin/env python3
"""
Oddsy Rolling Simulation - Phase 1 (Priorité Max)
------------------------------------------------
Simule une saison complète en rolling match par match.
Optimisé pour proxy 2025-26 basé sur rolling 2024-25.

Usage:
    python rolling_simulation_oddsy.py \
        --data data/processed/v13_xg_safe_features.csv \
        --season "2024-2025" \
        --model models/v23_retrained_2025_09_11_154613.joblib \
        --out results/rolling_2024_25 \
        --n_boot 500
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings('ignore')

# Features v2.3 validées (10 features optimales)
FEATURES_V23 = [
    "form_diff_normalized", "elo_diff_normalized", "h2h_score",
    "matchday_normalized", "shots_diff_normalized", "corners_diff_normalized", 
    "market_entropy_norm", "home_xg_eff_10", "away_goals_sum_5", "away_xg_eff_10"
]

def encode_target(result):
    """Convertit FullTimeResult (H/D/A) vers codes numériques (0/1/2)"""
    mapping = {'H': 0, 'D': 1, 'A': 2}
    return mapping[result]

def decode_target(code):
    """Convertit codes numériques (0/1/2) vers FullTimeResult (H/D/A)"""
    mapping = {0: 'H', 1: 'D', 2: 'A'}
    return mapping[code]

def bootstrap_accuracy(preds, trues, n_boot=500, seed=42):
    """Calcule accuracy bootstrapée avec intervalle de confiance 95%"""
    if n_boot == 0:
        acc = accuracy_score(trues, preds)
        return acc, acc, acc
    
    rng = np.random.default_rng(seed)
    n = len(preds)
    accs = []
    
    for _ in trange(n_boot, leave=False, desc="Bootstrap"):
        idx = rng.integers(0, n, n)
        boot_acc = accuracy_score(np.array(trues)[idx], np.array(preds)[idx])
        accs.append(boot_acc)
    
    mean_acc = np.mean(accs)
    lower_ci, upper_ci = np.percentile(accs, [2.5, 97.5])
    return mean_acc, lower_ci, upper_ci

def analyze_draw_performance(preds, trues):
    """Analyse spécifique des performances Draw (code 1)"""
    # Convert to numpy for easier indexing
    preds = np.array(preds)
    trues = np.array(trues)
    
    # Draw analysis
    draw_indices = (trues == 1)
    draw_recall = np.sum((preds == 1) & (trues == 1)) / np.sum(trues == 1) if np.sum(trues == 1) > 0 else 0
    draw_precision = np.sum((preds == 1) & (trues == 1)) / np.sum(preds == 1) if np.sum(preds == 1) > 0 else 0
    
    return {
        "draw_count": int(np.sum(trues == 1)),
        "draw_predicted": int(np.sum(preds == 1)),
        "draw_correct": int(np.sum((preds == 1) & (trues == 1))),
        "draw_recall": draw_recall,
        "draw_precision": draw_precision
    }

def rolling_simulation(df, model, season, n_boot=0, verbose=True):
    """
    Effectue simulation rolling match par match pour une saison.
    
    Returns:
        dict: Résultats complets avec métriques et prédictions
    """
    # Filter and sort season
    season_df = df[df["Season"] == season].copy()
    season_df = season_df.sort_values("Date").reset_index(drop=True)
    
    if len(season_df) == 0:
        raise ValueError(f"Aucun match trouvé pour la saison {season}")
    
    if verbose:
        print(f"🎯 Rolling simulation saison {season}")
        print(f"   Matches: {len(season_df)}")
        print(f"   Période: {season_df['Date'].iloc[0]} → {season_df['Date'].iloc[-1]}")
        print(f"   Bootstrap: {n_boot} itérations")
    
    # Rolling predictions
    predictions = []
    ground_truth = []
    match_details = []
    
    progress_bar = tqdm(season_df.iterrows(), total=len(season_df), desc="Rolling predictions") if verbose else season_df.iterrows()
    
    for idx, row in progress_bar:
        # Extract features
        X = row[FEATURES_V23].values.reshape(1, -1)
        
        # Predict
        pred_code = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        true_code = encode_target(row["FullTimeResult"])
        
        # Store results
        predictions.append(pred_code)
        ground_truth.append(true_code)
        
        match_details.append({
            "date": row["Date"],
            "home_team": row["HomeTeam"],
            "away_team": row["AwayTeam"],
            "true_result": row["FullTimeResult"],
            "true_code": true_code,
            "pred_code": pred_code,
            "pred_result": decode_target(pred_code),
            "correct": pred_code == true_code,
            "prob_home": pred_proba[0],
            "prob_draw": pred_proba[1], 
            "prob_away": pred_proba[2],
            "matchday": row.get("matchday_normalized", 0) * 38  # Approximation
        })
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    
    # Bootstrap if requested
    if n_boot > 0:
        boot_mean, boot_lower, boot_upper = bootstrap_accuracy(predictions, ground_truth, n_boot)
        bootstrap_results = {
            "mean_accuracy": boot_mean,
            "ci95_lower": boot_lower,
            "ci95_upper": boot_upper,
            "n_bootstrap": n_boot
        }
    else:
        bootstrap_results = None
    
    # Classification report
    target_names = ['Home', 'Draw', 'Away']
    class_report = classification_report(ground_truth, predictions, target_names=target_names, output_dict=True)
    
    # Draw-specific analysis
    draw_analysis = analyze_draw_performance(predictions, ground_truth)
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    
    if verbose:
        print(f"✅ Accuracy: {accuracy:.4f}")
        if bootstrap_results:
            print(f"   IC 95%: [{bootstrap_results['ci95_lower']:.4f}, {bootstrap_results['ci95_upper']:.4f}]")
        print(f"   Draw Recall: {draw_analysis['draw_recall']:.3f}")
    
    return {
        "season": season,
        "n_matches": len(season_df),
        "accuracy": accuracy,
        "bootstrap": bootstrap_results,
        "classification_report": class_report,
        "draw_analysis": draw_analysis,
        "confusion_matrix": cm.tolist(),
        "match_predictions": match_details
    }

def save_results(results, output_dir):
    """Sauvegarde les résultats dans différents formats"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON complet
    with open(output_path / "rolling_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # 2. CSV des prédictions
    predictions_df = pd.DataFrame(results["match_predictions"])
    predictions_df.to_csv(output_path / "match_predictions.csv", index=False)
    
    # 3. Résumé texte
    with open(output_path / "summary.txt", "w") as f:
        f.write(f"Rolling Simulation - Saison {results['season']}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Matches: {results['n_matches']}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        
        if results["bootstrap"]:
            boot = results["bootstrap"]
            f.write(f"Bootstrap Mean: {boot['mean_accuracy']:.4f}\n")
            f.write(f"IC 95%: [{boot['ci95_lower']:.4f}, {boot['ci95_upper']:.4f}]\n")
        
        draw = results["draw_analysis"]
        f.write(f"\nDraw Performance:\n")
        f.write(f"  Count: {draw['draw_count']}\n")
        f.write(f"  Predicted: {draw['draw_predicted']}\n")
        f.write(f"  Correct: {draw['draw_correct']}\n")
        f.write(f"  Recall: {draw['draw_recall']:.3f}\n")
        f.write(f"  Precision: {draw['draw_precision']:.3f}\n")
    
    print(f"💾 Résultats sauvegardés dans: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Oddsy Rolling Simulation")
    parser.add_argument("--data", required=True, help="Dataset CSV path")
    parser.add_argument("--season", required=True, help="Saison à analyser (ex: '2024-2025')")
    parser.add_argument("--model", required=True, help="Model .joblib path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--n_boot", type=int, default=0, help="Nombre de bootstrap (0=disabled)")
    parser.add_argument("--verbose", action="store_true", default=True, help="Mode verbose")
    
    args = parser.parse_args()
    
    print("🚀 Oddsy Rolling Simulation")
    print(f"   Data: {args.data}")
    print(f"   Season: {args.season}")
    print(f"   Model: {args.model}")
    print(f"   Output: {args.out}")
    
    # Load data and model
    try:
        df = pd.read_csv(args.data)
        model = joblib.load(args.model)
        print(f"✅ Data loaded: {len(df)} matches")
        print(f"✅ Model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Error loading data/model: {e}")
        return 1
    
    # Verify features
    missing_features = [f for f in FEATURES_V23 if f not in df.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        return 1
    
    # Run simulation
    try:
        results = rolling_simulation(df, model, args.season, args.n_boot, args.verbose)
        save_results(results, args.out)
        print("🎉 Rolling simulation terminée avec succès!")
        return 0
    except Exception as e:
        print(f"❌ Erreur simulation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())