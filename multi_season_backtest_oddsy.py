#!/usr/bin/env python3
"""
Oddsy Multi-Season Backtest avec Courbes Cumulatives
---------------------------------------------------
Backtest automatique multi-saisons avec bootstrap et visualisations.
Génère courbes accuracy cumulative match par match.

Usage:
    python multi_season_backtest_oddsy.py \
        --data data/processed/v13_xg_safe_features.csv \
        --model models/v23_retrained_2025_09_11_154613.joblib \
        --seasons "2019-2020" "2020-2021" "2021-2022" "2022-2023" "2023-2024" "2024-2025" \
        --out_dir results/backtest_all_seasons \
        --n_boot 500
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings('ignore')

# Import functions from rolling simulation
from rolling_simulation_oddsy import (
    FEATURES_V23, encode_target, decode_target, 
    bootstrap_accuracy, analyze_draw_performance
)

def rolling_simulation_light(df, model, season, verbose=False):
    """Version allégée du rolling simulation pour backtest multi-saisons"""
    # Filter and sort season
    season_df = df[df["Season"] == season].copy()
    season_df = season_df.sort_values("Date").reset_index(drop=True)
    
    if len(season_df) == 0:
        return None
    
    # Rolling predictions
    predictions = []
    ground_truth = []
    cumulative_accuracy = []
    
    progress_desc = f"Rolling {season}" if verbose else None
    iterator = tqdm(season_df.iterrows(), total=len(season_df), desc=progress_desc, leave=False) if verbose else season_df.iterrows()
    
    correct_count = 0
    for match_idx, (_, row) in enumerate(iterator, 1):
        # Extract features and predict
        X = row[FEATURES_V23].values.reshape(1, -1)
        pred_code = model.predict(X)[0]
        true_code = encode_target(row["FullTimeResult"])
        
        # Track predictions
        predictions.append(pred_code)
        ground_truth.append(true_code)
        
        # Update cumulative accuracy
        if pred_code == true_code:
            correct_count += 1
        cumulative_accuracy.append(correct_count / match_idx)
    
    return {
        "season": season,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "cumulative_accuracy": cumulative_accuracy,
        "n_matches": len(season_df)
    }

def generate_cumulative_plot(season_results, output_dir):
    """Génère courbe accuracy cumulative pour toutes les saisons"""
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(season_results)))
    
    for i, result in enumerate(season_results):
        if result is None:
            continue
            
        matches = range(1, result["n_matches"] + 1)
        cumulative_acc = result["cumulative_accuracy"]
        
        plt.plot(matches, cumulative_acc, 
                label=f'{result["season"]} (final: {cumulative_acc[-1]:.3f})',
                color=colors[i], linewidth=2, alpha=0.8)
    
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% target')
    plt.axhline(y=0.55, color='orange', linestyle='--', alpha=0.7, label='55% excellent')
    plt.axhline(y=1/3, color='gray', linestyle=':', alpha=0.7, label='33% random')
    
    plt.xlabel('Match Number')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Rolling Accuracy Evolution - Multi-Season Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "cumulative_accuracy_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Courbe cumulative sauvegardée: {output_path}")

def generate_season_comparison_plot(backtest_summary, output_dir):
    """Génère graphique comparaison par saison avec IC bootstrap"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy par saison avec IC
    seasons = backtest_summary["season"].tolist()
    accuracies = backtest_summary["accuracy"].tolist()
    
    if "ci95_lower" in backtest_summary.columns:
        lower_ci = backtest_summary["ci95_lower"].tolist()
        upper_ci = backtest_summary["ci95_upper"].tolist()
        errors = [[acc - low for acc, low in zip(accuracies, lower_ci)],
                  [up - acc for up, acc in zip(upper_ci, accuracies)]]
    else:
        errors = None
    
    bars = ax1.bar(range(len(seasons)), accuracies, 
                   yerr=errors, capsize=5, alpha=0.7, color='steelblue')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% target')
    ax1.axhline(y=0.55, color='orange', linestyle='--', alpha=0.7, label='55% excellent')
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Season (with 95% CI)')
    ax1.set_xticks(range(len(seasons)))
    ax1.set_xticklabels(seasons, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution accuracy
    ax2.hist(accuracies, bins=8, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(np.mean(accuracies), color='red', linestyle='-', linewidth=2, 
                label=f'Moyenne: {np.mean(accuracies):.3f}')
    ax2.axvline(np.mean(accuracies) - np.std(accuracies), color='orange', 
                linestyle='--', alpha=0.7, label=f'±1σ: {np.std(accuracies):.3f}')
    ax2.axvline(np.mean(accuracies) + np.std(accuracies), color='orange', 
                linestyle='--', alpha=0.7)
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Count')
    ax2.set_title('Accuracy Distribution Across Seasons')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "season_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparaison saisons sauvegardée: {output_path}")

def run_multi_season_backtest(df, model, seasons, n_boot=500, verbose=True):
    """Exécute backtest sur plusieurs saisons avec bootstrap"""
    
    if verbose:
        print(f"🔬 Multi-Season Backtest")
        print(f"   Saisons: {len(seasons)}")
        print(f"   Bootstrap: {n_boot} itérations par saison")
    
    # Rolling simulation pour chaque saison
    season_results = []
    backtest_summary = []
    
    for season in tqdm(seasons, desc="Saisons", disable=not verbose):
        # Rolling simulation
        result = rolling_simulation_light(df, model, season, verbose=verbose)
        if result is None:
            if verbose:
                print(f"⚠️  Pas de données pour {season}")
            continue
        
        season_results.append(result)
        
        # Métriques de base
        preds = result["predictions"]
        trues = result["ground_truth"]
        accuracy = accuracy_score(trues, preds)
        
        # Bootstrap si demandé
        if n_boot > 0:
            boot_mean, boot_lower, boot_upper = bootstrap_accuracy(preds, trues, n_boot)
            bootstrap_data = {
                "boot_mean": boot_mean,
                "ci95_lower": boot_lower,
                "ci95_upper": boot_upper
            }
        else:
            bootstrap_data = {}
        
        # Analyse Draw
        draw_analysis = analyze_draw_performance(preds, trues)
        
        # Summary pour cette saison
        season_summary = {
            "season": season,
            "n_matches": result["n_matches"],
            "accuracy": accuracy,
            "final_cumulative": result["cumulative_accuracy"][-1],
            "draw_recall": draw_analysis["draw_recall"],
            **bootstrap_data
        }
        
        backtest_summary.append(season_summary)
        
        if verbose:
            ic_str = f" (IC: [{bootstrap_data.get('ci95_lower', 0):.3f}, {bootstrap_data.get('ci95_upper', 0):.3f}])" if bootstrap_data else ""
            print(f"   {season}: {accuracy:.4f}{ic_str}")
    
    # Convert to DataFrame
    backtest_df = pd.DataFrame(backtest_summary)
    
    # Statistiques globales
    if len(backtest_df) > 0:
        global_stats = {
            "mean_accuracy": backtest_df["accuracy"].mean(),
            "std_accuracy": backtest_df["accuracy"].std(),
            "min_accuracy": backtest_df["accuracy"].min(),
            "max_accuracy": backtest_df["accuracy"].max(),
            "mean_draw_recall": backtest_df["draw_recall"].mean(),
            "n_seasons_analyzed": len(backtest_df)
        }
        
        if verbose:
            print(f"\n📈 Statistiques globales:")
            print(f"   Accuracy moyenne: {global_stats['mean_accuracy']:.4f} ± {global_stats['std_accuracy']:.4f}")
            print(f"   Range: [{global_stats['min_accuracy']:.4f}, {global_stats['max_accuracy']:.4f}]")
            print(f"   Draw recall moyen: {global_stats['mean_draw_recall']:.3f}")
    else:
        global_stats = {}
    
    return {
        "season_results": season_results,
        "backtest_summary": backtest_df,
        "global_stats": global_stats
    }

def save_backtest_results(results, output_dir):
    """Sauvegarde les résultats du backtest multi-saisons"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. CSV summary des saisons
    results["backtest_summary"].to_csv(output_path / "multi_season_summary.csv", index=False)
    
    # 2. JSON complet
    # Prepare data for JSON serialization
    json_data = {
        "global_stats": results["global_stats"],
        "backtest_summary": results["backtest_summary"].to_dict('records'),
        "season_details": []
    }
    
    for season_result in results["season_results"]:
        if season_result:
            json_data["season_details"].append({
                "season": season_result["season"],
                "n_matches": season_result["n_matches"],
                "cumulative_accuracy": season_result["cumulative_accuracy"]
            })
    
    with open(output_path / "multi_season_backtest.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    
    # 3. Rapport texte
    with open(output_path / "backtest_report.txt", "w") as f:
        f.write("Multi-Season Backtest Report\n")
        f.write("=" * 40 + "\n\n")
        
        if results["global_stats"]:
            stats = results["global_stats"]
            f.write(f"Global Statistics:\n")
            f.write(f"  Seasons analyzed: {stats['n_seasons_analyzed']}\n")
            f.write(f"  Mean accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}\n")
            f.write(f"  Range: [{stats['min_accuracy']:.4f}, {stats['max_accuracy']:.4f}]\n")
            f.write(f"  Mean draw recall: {stats['mean_draw_recall']:.3f}\n\n")
        
        f.write("Season Details:\n")
        for _, row in results["backtest_summary"].iterrows():
            f.write(f"  {row['season']}: {row['accuracy']:.4f}")
            if 'ci95_lower' in row:
                f.write(f" (IC: [{row['ci95_lower']:.3f}, {row['ci95_upper']:.3f}])")
            f.write(f" - {row['n_matches']} matches\n")
    
    # 4. Générer visualisations
    generate_cumulative_plot(results["season_results"], output_path)
    generate_season_comparison_plot(results["backtest_summary"], output_path)
    
    print(f"💾 Backtest results sauvegardés dans: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Oddsy Multi-Season Backtest")
    parser.add_argument("--data", required=True, help="Dataset CSV path")
    parser.add_argument("--model", required=True, help="Model .joblib path")
    parser.add_argument("--seasons", nargs="+", required=True, help="Saisons à analyser")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--n_boot", type=int, default=500, help="Nombre de bootstrap par saison")
    parser.add_argument("--verbose", action="store_true", default=True, help="Mode verbose")
    
    args = parser.parse_args()
    
    print("🔬 Oddsy Multi-Season Backtest")
    print(f"   Data: {args.data}")
    print(f"   Saisons: {args.seasons}")
    print(f"   Model: {args.model}")
    print(f"   Output: {args.out_dir}")
    print(f"   Bootstrap: {args.n_boot}")
    
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
    
    # Run backtest
    try:
        results = run_multi_season_backtest(df, model, args.seasons, args.n_boot, args.verbose)
        save_backtest_results(results, args.out_dir)
        print("🎉 Multi-season backtest terminé avec succès!")
        return 0
    except Exception as e:
        print(f"❌ Erreur backtest: {e}")
        return 1

if __name__ == "__main__":
    exit(main())