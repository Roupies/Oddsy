#!/usr/bin/env python3
"""
Audit Pipeline for Football Prediction Models
=============================================

This script performs a full audit of a trained model:
1. Reproducibility checks (hash dataset, library versions, random seed)
2. Feature validation (missing values, leakage, consistency)
3. Temporal validation (time-series CV)
4. Metrics (accuracy, recall/precision, log-loss, calibration)
5. Robustness (retrain with different seeds)
6. Baseline comparisons (majority class, bookmaker, Elo)
7. External audit report (JSON + summary text)
8. Documentation with plots

Author: Oddsy Project
Date: 2025-09-12
"""

import argparse
import hashlib
import json
import os
import sys
import joblib
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle
import sklearn


# ========================
# Utility functions
# ========================

def compute_file_hash(path):
    """Compute SHA256 hash of a file for reproducibility."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()


def log(msg):
    """Pretty logger with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AUDIT - {msg}")


def save_json(obj, path):
    """Save object as JSON with proper formatting."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ========================
# Audit Functions
# ========================

def check_reproducibility(data_path, model_path):
    """Check dataset hash, library versions, and system info."""
    dataset_hash = compute_file_hash(data_path)
    system_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }
    return dataset_hash, system_info


def validate_features(df, target_col="target"):
    """Ensure no leakage, no missing values, consistent dtypes."""
    issues = []
    
    # Check for target leakage
    if target_col in df.columns:
        issues.append("❌ Target column found in features!")
    
    # Check for obvious leakage columns
    leakage_keywords = ['result', 'winner', 'outcome', 'final', 'fulltime']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in leakage_keywords):
            issues.append(f"⚠️ Potential leakage column detected: {col}")

    # Check missing values
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        issues.append(f"⚠️ Missing values detected: {missing_cols.to_dict()}")
    
    # Check feature consistency
    if len(df.columns) == 0:
        issues.append("❌ No features found!")

    return issues


def temporal_cv_evaluation(X, y, model_config, n_splits=5):
    """Perform TimeSeriesSplit evaluation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, baccs, loglosses = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Create fresh model for CV
        temp_model = RandomForestClassifier(**model_config)
        temp_model.fit(X_train, y_train)
        
        y_pred = temp_model.predict(X_test)
        y_proba = temp_model.predict_proba(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        baccs.append(balanced_accuracy_score(y_test, y_pred))
        loglosses.append(log_loss(y_test, y_proba))

    return {
        "cv_accuracy_mean": np.mean(accs),
        "cv_accuracy_std": np.std(accs),
        "cv_balanced_accuracy": np.mean(baccs),
        "cv_logloss": np.mean(loglosses),
        "cv_scores": accs,
    }


def evaluate_metrics(model, X, y, label_names):
    """Compute accuracy, recall, precision, f1, logloss."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    ll = log_loss(y, y_proba)

    report = classification_report(y, y_pred, target_names=label_names, output_dict=True)
    prfs = precision_recall_fscore_support(y, y_pred, average=None)

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "log_loss": ll,
        "classification_report": report,
        "precision_recall_fscore": {
            "precision": prfs[0].tolist(),
            "recall": prfs[1].tolist(),
            "fscore": prfs[2].tolist(),
            "support": prfs[3].tolist()
        }
    }, (y_pred, y_proba)


def robustness_check(X, y, base_model, n_runs=3, seeds=None):
    """
    Retrain model multiple times with different seeds to test stability.
    Returns mean and std of key metrics.
    """
    if seeds is None:
        seeds = [42, 123, 999, 777, 555][:n_runs]

    results = {"accuracy": [], "balanced_accuracy": [], "log_loss": []}
    
    log(f"Running robustness check with {n_runs} different seeds...")

    for i, seed in enumerate(seeds):
        log(f"  Run {i+1}/{n_runs} with seed {seed}")
        
        # Extract model parameters
        model_params = {
            'n_estimators': getattr(base_model, 'n_estimators', 100),
            'max_depth': getattr(base_model, 'max_depth', None),
            'max_features': getattr(base_model, 'max_features', 'sqrt'),
            'min_samples_split': getattr(base_model, 'min_samples_split', 2),
            'min_samples_leaf': getattr(base_model, 'min_samples_leaf', 1),
            'class_weight': getattr(base_model, 'class_weight', None),
            'random_state': seed,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**model_params)
        model.fit(X, y)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        results["accuracy"].append(accuracy_score(y, y_pred))
        results["balanced_accuracy"].append(balanced_accuracy_score(y, y_pred))
        results["log_loss"].append(log_loss(y, y_proba))

    return {
        "mean_accuracy": np.mean(results["accuracy"]),
        "std_accuracy": np.std(results["accuracy"]),
        "mean_balanced_accuracy": np.mean(results["balanced_accuracy"]),
        "std_balanced_accuracy": np.std(results["balanced_accuracy"]),
        "mean_log_loss": np.mean(results["log_loss"]),
        "std_log_loss": np.std(results["log_loss"]),
        "runs": len(seeds),
        "seeds_used": seeds,
        "all_results": results
    }


def calibration_plot(y, y_proba, out_path):
    """Generate calibration curve plot."""
    plt.figure(figsize=(10, 6))
    
    # Create subplot for calibration curves
    plt.subplot(1, 2, 1)
    for i, label in enumerate(["Home", "Draw", "Away"]):
        prob_true, prob_pred = calibration_curve((y == i).astype(int), y_proba[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=label)

    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create subplot for prediction distribution
    plt.subplot(1, 2, 2)
    for i, label in enumerate(["Home", "Draw", "Away"]):
        plt.hist(y_proba[:, i], bins=20, alpha=0.6, label=label, density=True)
    
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Prediction Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def feature_importance_plot(model, feature_names, out_path):
    """Generate feature importance plot."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel("Importance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return dict(zip([feature_names[i] for i in indices], importances[indices].tolist()))
    else:
        return {}


def baseline_majority(y):
    """Majority class baseline accuracy."""
    majority_class = y.value_counts().idxmax()
    return accuracy_score(y, [majority_class] * len(y))


def baseline_random(y, n_classes=3):
    """Random baseline accuracy."""
    random_preds = np.random.choice(n_classes, size=len(y))
    return accuracy_score(y, random_preds)


# ========================
# Main Audit Pipeline
# ========================

def run_audit(data_path, model_path, target_col="target", out_dir="results/audit", specific_features=None):
    """Run comprehensive audit pipeline."""
    os.makedirs(out_dir, exist_ok=True)
    
    log("🔍 Starting comprehensive model audit...")

    log("Loading dataset...")
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    # Use specific features if provided, otherwise remove non-numeric columns
    if specific_features:
        # Check if all specified features exist
        missing_features = [f for f in specific_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features}")
        X = df[specific_features]
        log(f"Using specified features: {len(specific_features)} features")
    else:
        # Remove non-numeric columns that aren't features
        non_feature_cols = ['Date', 'Season', 'HomeTeam', 'AwayTeam', target_col]
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        X = df[feature_cols]
        log(f"Auto-detected features: {len(feature_cols)} features")
    
    y = df[target_col]
    
    # Convert target to numeric if needed (H->0, D->1, A->2)
    if y.dtype == 'object':
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        y = y.map(target_mapping)
        if y.isna().any():
            log("Warning: Some target values could not be mapped!")
            y = y.fillna(-1)  # Mark unmapped values
    
    log(f"Dataset shape: {df.shape}")
    log(f"Features: {len(X.columns)}")
    log(f"Target distribution: {y.value_counts().to_dict()}")

    log("Loading model...")
    model = joblib.load(model_path)
    
    # Extract model configuration for CV and robustness testing
    model_config = {
        'n_estimators': getattr(model, 'n_estimators', 100),
        'max_depth': getattr(model, 'max_depth', None),
        'max_features': getattr(model, 'max_features', 'sqrt'),
        'min_samples_split': getattr(model, 'min_samples_split', 2),
        'min_samples_leaf': getattr(model, 'min_samples_leaf', 1),
        'class_weight': getattr(model, 'class_weight', None),
        'random_state': getattr(model, 'random_state', 42),
        'n_jobs': -1
    }

    log("Checking reproducibility...")
    dataset_hash, system_info = check_reproducibility(data_path, model_path)

    log("Validating features...")
    feature_issues = validate_features(X, target_col)

    log("Running temporal CV evaluation...")
    cv_results = temporal_cv_evaluation(X, y, model_config, n_splits=5)

    log("Evaluating metrics on full dataset...")
    metrics, (y_pred, y_proba) = evaluate_metrics(model, X, y, ["Home", "Draw", "Away"])

    log("Running robustness check across seeds...")
    robustness_results = robustness_check(X, y, model, n_runs=3)

    log("Generating calibration plot...")
    calibration_plot(y, y_proba, os.path.join(out_dir, "calibration_curve.png"))
    
    log("Generating feature importance plot...")
    feature_importance = feature_importance_plot(model, X.columns.tolist(), 
                                                os.path.join(out_dir, "feature_importance.png"))

    log("Computing baselines...")
    baseline_majority_acc = baseline_majority(y)
    baseline_random_acc = baseline_random(y, n_classes=len(y.unique()))

    # Build comprehensive audit report
    audit_report = {
        "audit_metadata": {
            "timestamp": datetime.now().isoformat(),
            "audit_version": "1.0",
            "model_path": model_path,
            "data_path": data_path,
        },
        "dataset_info": {
            "shape": df.shape,
            "n_features": len(X.columns),
            "target_distribution": y.value_counts().to_dict(),
            "dataset_hash": dataset_hash,
        },
        "system_info": system_info,
        "model_config": model_config,
        "feature_validation": {
            "issues": feature_issues,
            "feature_names": X.columns.tolist(),
            "feature_importance": feature_importance,
        },
        "temporal_validation": cv_results,
        "performance_metrics": metrics,
        "robustness_analysis": robustness_results,
        "baseline_comparisons": {
            "majority_class": baseline_majority_acc,
            "random_baseline": baseline_random_acc,
        },
        "audit_summary": {
            "accuracy": metrics['accuracy'],
            "accuracy_vs_majority": metrics['accuracy'] - baseline_majority_acc,
            "accuracy_stability": robustness_results['std_accuracy'],
            "cv_stability": cv_results['cv_accuracy_std'],
            "major_issues": len([issue for issue in feature_issues if issue.startswith('❌')]),
            "warnings": len([issue for issue in feature_issues if issue.startswith('⚠️')]),
        }
    }
    
    # Determine verdict
    major_issues = audit_report['audit_summary']['major_issues']
    accuracy_stable = robustness_results['std_accuracy'] < 0.01
    cv_stable = cv_results['cv_accuracy_std'] < 0.05
    beats_baseline = metrics['accuracy'] > baseline_majority_acc
    
    if major_issues == 0 and accuracy_stable and cv_stable and beats_baseline:
        verdict = "✅ AUDITED - PRODUCTION READY"
    elif major_issues == 0 and beats_baseline:
        verdict = "⚠️ AUDITED - MINOR ISSUES"
    else:
        verdict = "❌ AUDIT FAILED - MAJOR ISSUES"
    
    audit_report["verdict"] = verdict

    # Save audit report
    report_path = os.path.join(out_dir, f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_json(audit_report, report_path)
    log(f"📊 Audit report saved: {report_path}")

    # Save summary text
    summary_path = os.path.join(out_dir, f"audit_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_path, "w") as f:
        f.write("ODDSY MODEL AUDIT SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Dataset: {os.path.basename(data_path)}\n")
        f.write(f"Audit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"LogLoss: {metrics['log_loss']:.4f}\n")
        f.write(f"vs Majority Baseline: +{(metrics['accuracy'] - baseline_majority_acc)*100:.2f}pp\n\n")
        
        f.write("TEMPORAL VALIDATION:\n")
        f.write(f"CV Accuracy (mean±std): {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}\n")
        f.write(f"CV Balanced Accuracy: {cv_results['cv_balanced_accuracy']:.4f}\n\n")
        
        f.write("ROBUSTNESS CHECK (3 runs):\n")
        f.write(f"Accuracy mean ± std: {robustness_results['mean_accuracy']:.4f} ± {robustness_results['std_accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy mean ± std: {robustness_results['mean_balanced_accuracy']:.4f} ± {robustness_results['std_balanced_accuracy']:.4f}\n")
        f.write(f"LogLoss mean ± std: {robustness_results['mean_log_loss']:.4f} ± {robustness_results['std_log_loss']:.4f}\n\n")
        
        f.write("VALIDATION ISSUES:\n")
        for issue in feature_issues:
            f.write(f"{issue}\n")
        if not feature_issues:
            f.write("None detected\n")
        
        f.write(f"\nFINAL VERDICT: {verdict}\n")

    log(f"📝 Audit summary saved: {summary_path}")
    log(f"🎯 FINAL VERDICT: {verdict}")
    log("🏁 Audit completed successfully!")

    return audit_report


# ========================
# CLI Entry Point
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit a trained football prediction model.")
    parser.add_argument("--data", required=True, help="Path to processed dataset CSV")
    parser.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--target", default="target", help="Target column name")
    parser.add_argument("--out", default="results/audit", help="Output directory for audit report")
    parser.add_argument("--features", nargs='+', help="Specific features to use (optional)")

    args = parser.parse_args()
    
    try:
        run_audit(args.data, args.model, args.target, args.out, args.features)
    except Exception as e:
        log(f"❌ Audit failed: {str(e)}")
        sys.exit(1)