import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def load_model_and_data():
    """Load the latest trained model and data"""
    # Find latest model
    model_files = [f for f in os.listdir('models/') if f.startswith('random_forest_') and f.endswith('.joblib')]
    if not model_files:
        raise FileNotFoundError("No trained model found. Run train_model.py first.")
    
    latest_model = sorted(model_files)[-1]
    model_path = f"models/{latest_model}"
    
    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)
    
    # Load data
    X = pd.read_csv('data/processed/X_features.csv')
    y = pd.read_csv('data/processed/y_target.csv')['target']
    
    # Load configuration
    with open('config/target_mapping.json', 'r') as f:
        target_info = json.load(f)
    
    return model, X, y, target_info, latest_model

def comprehensive_evaluation(model, X, y, target_info):
    """Comprehensive model evaluation with all metrics"""
    print("=== Comprehensive Model Evaluation ===")
    
    # Basic predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    # Basic metrics
    accuracy = accuracy_score(y, y_pred)
    precision_macro = precision_score(y, y_pred, average='macro')
    precision_weighted = precision_score(y, y_pred, average='weighted')
    recall_macro = recall_score(y, y_pred, average='macro')
    recall_weighted = recall_score(y, y_pred, average='weighted')
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    
    print(f"\n--- Overall Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Precision (Weighted): {precision_weighted:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"Recall (Weighted): {recall_weighted:.4f}")
    print(f"F1 (Macro): {f1_macro:.4f}")
    print(f"F1 (Weighted): {f1_weighted:.4f}")
    
    # Per-class metrics
    print(f"\n--- Per-Class Report ---")
    class_names = target_info['class_names']
    report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\n--- Confusion Matrix ---")
    print("Predicted ->")
    print("Actual ↓   ", " ".join(f"{name:>8}" for name in class_names))
    for i, name in enumerate(class_names):
        print(f"{name:>8}   ", " ".join(f"{cm[i,j]:>8}" for j in range(len(class_names))))
    
    # Class-wise accuracy
    print(f"\n--- Class-wise Accuracy ---")
    for i, name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        print(f"{name}: {class_acc:.4f} ({cm[i, i]}/{cm[i, :].sum()})")
    
    # Baseline comparisons
    print(f"\n--- Baseline Comparisons ---")
    random_acc = 1/3
    majority_class = y.mode()[0]
    majority_acc = (y == majority_class).mean()
    
    print(f"Random (33.33%): {'✓ BEAT' if accuracy > random_acc else '✗ BELOW'}")
    print(f"Majority Class ({majority_acc:.3f}): {'✓ BEAT' if accuracy > majority_acc else '✗ BELOW'}")
    print(f"Target 50%: {'✓ ACHIEVED' if accuracy > 0.5 else '✗ BELOW'}")
    print(f"Target 55%: {'✓ EXCELLENT' if accuracy > 0.55 else '✗ BELOW'}")
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': y_pred.tolist(),
        'probabilities': y_prob.tolist()
    }

def cross_validation_evaluation(model, X, y, cv_folds=5):
    """Time Series Cross-Validation evaluation"""
    print(f"\n=== Time Series Cross-Validation ({cv_folds} folds) ===")
    
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=tscv, scoring=scoring, n_jobs=-1)
    
    # Display results
    metrics_summary = {}
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        mean_score = scores.mean()
        std_score = scores.std()
        metrics_summary[metric] = {
            'mean': mean_score,
            'std': std_score,
            'scores': scores.tolist()
        }
        print(f"{metric.title()}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        print(f"  Fold scores: {scores}")
    
    return metrics_summary

def create_visualizations(y_true, y_pred, target_info, model_name):
    """Create evaluation visualizations"""
    print("\n=== Creating Visualizations ===")
    
    os.makedirs('evaluation/', exist_ok=True)
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    class_names = target_info['class_names']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    timestamp = datetime.now().strftime("%Y_%m_%d")
    plt.tight_layout()
    plt.savefig(f'evaluation/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Class Distribution vs Predictions
    plt.figure(figsize=(12, 4))
    
    # True distribution
    plt.subplot(1, 2, 1)
    true_dist = pd.Series(y_true).value_counts().sort_index()
    true_dist.plot(kind='bar', color='lightblue')
    plt.title('True Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([0, 1, 2], class_names, rotation=0)
    
    # Predicted distribution
    plt.subplot(1, 2, 2)
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    pred_dist.plot(kind='bar', color='lightcoral')
    plt.title('Predicted Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([0, 1, 2], class_names, rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'evaluation/class_distributions_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved:")
    print(f"- Confusion matrix: evaluation/confusion_matrix_{timestamp}.png")
    print(f"- Class distributions: evaluation/class_distributions_{timestamp}.png")
    
    return timestamp

def save_evaluation_results(results, cv_results, target_info, model_name, timestamp):
    """Save comprehensive evaluation results"""
    
    # Compile all results
    evaluation_report = {
        'model_name': model_name,
        'evaluation_date': timestamp,
        'target_mapping': target_info,
        'overall_metrics': {
            'accuracy': results['accuracy'],
            'precision_macro': results['precision_macro'],
            'precision_weighted': results['precision_weighted'],
            'recall_macro': results['recall_macro'],
            'recall_weighted': results['recall_weighted'],
            'f1_macro': results['f1_macro'],
            'f1_weighted': results['f1_weighted']
        },
        'confusion_matrix': results['confusion_matrix'],
        'classification_report': results['classification_report'],
        'cross_validation': cv_results,
        'baseline_comparisons': {
            'random_baseline': 0.333,
            'majority_baseline': max(target_info['distribution'].values()) / sum(target_info['distribution'].values()),
            'beats_random': results['accuracy'] > 0.333,
            'beats_majority': results['accuracy'] > max(target_info['distribution'].values()) / sum(target_info['distribution'].values()),
            'achieves_50_percent': results['accuracy'] > 0.5,
            'achieves_55_percent': results['accuracy'] > 0.55
        }
    }
    
    # Save to JSON
    output_file = f"evaluation/evaluation_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print(f"\nEvaluation results saved: {output_file}")
    
    return evaluation_report

if __name__ == "__main__":
    # Load model and data
    model, X, y, target_info, model_name = load_model_and_data()
    
    # Comprehensive evaluation
    results = comprehensive_evaluation(model, X, y, target_info)
    
    # Cross-validation
    cv_results = cross_validation_evaluation(model, X, y, cv_folds=5)
    
    # Create visualizations
    timestamp = create_visualizations(y, results['predictions'], target_info, model_name)
    
    # Save results
    report = save_evaluation_results(results, cv_results, target_info, model_name, timestamp)
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Model: {model_name}")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score (Macro): {results['f1_macro']:.4f}")
    print(f"Cross-Validation Accuracy: {cv_results['accuracy']['mean']:.4f} (+/- {cv_results['accuracy']['std']*2:.4f})")
    
    # Performance assessment
    if results['accuracy'] > 0.55:
        print("🏆 EXCELLENT: Model achieves > 55% accuracy!")
    elif results['accuracy'] > 0.50:
        print("✅ GOOD: Model achieves > 50% accuracy!")
    elif results['accuracy'] > 0.436:
        print("✅ ACCEPTABLE: Model beats majority baseline!")
    else:
        print("❌ NEEDS IMPROVEMENT: Model below majority baseline")
    
    print("Evaluation complete!")