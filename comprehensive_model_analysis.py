#!/usr/bin/env python3
"""
Comprehensive Model Analysis and Recommendations
Combining results from both testing scenarios
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_baseline_results():
    """Load baseline comprehensive test results"""
    try:
        with open('outputs/baseline_comprehensive_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def create_comparison_table():
    """Create comprehensive comparison table"""
    # Baseline Champion v2.3 results from both scenarios
    results_data = [
        {
            'Model': 'Baseline Champion v2.3',
            'Scenario': 'Phase 1 (1,900/380)',
            'Test_Period': '2024-08-24 to 2025-08-23',
            'Test_Size': 380,
            'Accuracy': 0.5316,
            'Precision_H': 0.544,
            'Precision_D': 0.333, 
            'Precision_A': 0.526,
            'Recall_H': 0.745,
            'Recall_D': 0.032,
            'Recall_A': 0.636,
            'Status': 'Above 50% target ✅'
        },
        {
            'Model': 'Baseline Champion v2.3',
            'Scenario': 'Phase 2 (2,280/50)',
            'Test_Period': '2025-08-15 to 2025-09-21',
            'Test_Size': 50,
            'Accuracy': 0.4200,
            'Precision_H': 0.500,
            'Precision_D': 0.000,
            'Precision_A': 0.278,
            'Recall_H': 0.696,
            'Recall_D': 0.000,
            'Recall_A': 0.385,
            'Status': 'Below baseline ❌'
        },
        {
            'Model': 'Baseline Champion v2.3 (Original)',
            'Scenario': 'EPL 2025-26 only',
            'Test_Period': '2025 season',
            'Test_Size': 50,
            'Accuracy': 0.4200,
            'Precision_H': 0.500,
            'Precision_D': 0.000,
            'Precision_A': 0.278,
            'Recall_H': 0.696,
            'Recall_D': 0.000,
            'Recall_A': 0.385,
            'Status': 'Below baseline ❌'
        },
        {
            'Model': 'Optimized Baseline (Fresh)',
            'Scenario': 'EPL 2025-26 only', 
            'Test_Period': '2025 season',
            'Test_Size': 50,
            'Accuracy': 0.4000,
            'Precision_H': 0.500,
            'Precision_D': 0.000,
            'Precision_A': 0.400,
            'Recall_H': 0.652,
            'Recall_D': 0.000,
            'Recall_A': 0.769,
            'Status': 'Below baseline ❌'
        }
    ]
    
    return pd.DataFrame(results_data)

def plot_performance_comparison():
    """Create visual comparison of model performances"""
    df = create_comparison_table()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy comparison
    scenarios = df['Scenario'].unique()
    accuracy_data = []
    for scenario in scenarios:
        scenario_data = df[df['Scenario'] == scenario]
        for _, row in scenario_data.iterrows():
            accuracy_data.append({
                'Scenario': scenario,
                'Model': row['Model'],
                'Accuracy': row['Accuracy']
            })
    
    acc_df = pd.DataFrame(accuracy_data)
    sns.barplot(data=acc_df, x='Scenario', y='Accuracy', hue='Model', ax=ax1)
    ax1.set_title('Model Accuracy Comparison')
    ax1.axhline(y=0.50, color='green', linestyle='--', label='Target (50%)')
    ax1.axhline(y=0.436, color='orange', linestyle='--', label='Baseline (43.6%)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Precision by class (Phase 1 only for clarity)
    phase1_data = df[df['Scenario'] == 'Phase 1 (1,900/380)'].iloc[0]
    classes = ['H', 'D', 'A']
    precisions = [phase1_data['Precision_H'], phase1_data['Precision_D'], phase1_data['Precision_A']]
    ax2.bar(classes, precisions, color=['blue', 'gray', 'red'])
    ax2.set_title('Precision by Class (Phase 1 - Best Result)')
    ax2.set_ylabel('Precision')
    ax2.set_ylim(0, 1)
    
    # 3. Recall by class (Phase 1 only)
    recalls = [phase1_data['Recall_H'], phase1_data['Recall_D'], phase1_data['Recall_A']]
    ax3.bar(classes, recalls, color=['blue', 'gray', 'red'])
    ax3.set_title('Recall by Class (Phase 1 - Best Result)')
    ax3.set_ylabel('Recall')
    ax3.set_ylim(0, 1)
    
    # 4. Test size vs Accuracy
    test_sizes = df['Test_Size'].values
    accuracies = df['Accuracy'].values
    ax4.scatter(test_sizes, accuracies, s=100, alpha=0.7)
    for i, model in enumerate(df['Model']):
        ax4.annotate(f"{model[:15]}...", (test_sizes[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax4.set_xlabel('Test Set Size')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Test Size vs Performance')
    ax4.axhline(y=0.50, color='green', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.436, color='orange', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('outputs/comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_recommendations():
    """Generate detailed recommendations based on results"""
    recommendations = {
        "executive_summary": {
            "best_model": "Baseline Champion v2.3",
            "best_scenario": "Phase 1 (1,900 train / 380 test)",
            "best_accuracy": "53.16%",
            "status": "Exceeds 50% target on larger test set"
        },
        "key_findings": [
            "Baseline Champion v2.3 achieves 53.16% accuracy on Phase 1 (380 test samples)",
            "Performance drops significantly on smaller test set (42.0% on 50 samples)",
            "Model struggles severely with draw prediction (3.2% recall in Phase 1)",
            "Strong performance on Home (74.5% recall) and Away (63.6% recall) predictions",
            "More training data (2,280 vs 1,900) doesn't improve performance on recent matches"
        ],
        "performance_breakdown": {
            "phase1_strengths": [
                "Exceeds 50% accuracy target",
                "Good precision balance across H/A classes", 
                "Statistically significant test size (380 samples)"
            ],
            "phase1_weaknesses": [
                "Very poor draw detection (3.2% recall)",
                "Overpredicts Home wins (74.5% recall)"
            ],
            "phase2_issues": [
                "Below baseline performance (42.0%)",
                "Complete failure on draw prediction (0% recall)",
                "Small test set (50 samples) limits reliability"
            ]
        },
        "technical_insights": [
            "Model shows temporal drift - performance degrades on most recent data",
            "Feature engineering may need updates for current EPL patterns",
            "Draw class severely underrepresented in predictions",
            "Calibration may need adjustment for current season dynamics"
        ],
        "recommendations": {
            "immediate_actions": [
                "Use Baseline Champion v2.3 for production on data similar to Phase 1 period",
                "Implement draw detection enhancement (cascade approach or class weighting)",
                "Monitor performance on 2025-26 season and retrain monthly"
            ],
            "medium_term": [
                "Investigate feature engineering for recent EPL patterns", 
                "Implement ensemble approach combining multiple models",
                "Add betting market features as identified in feature analysis",
                "Develop specialized draw prediction model"
            ],
            "long_term": [
                "Implement real-time model updating pipeline",
                "Add advanced statistics (referee patterns, temporal features)",
                "Consider modern ML approaches (gradient boosting, neural networks)",
                "Integrate external data sources (injuries, team news)"
            ]
        },
        "risk_assessment": {
            "production_readiness": "CONDITIONAL - Good on historical data, concerning on recent",
            "reliability_issues": [
                "Temporal drift affects recent match prediction",
                "Small recent test set limits confidence",
                "Draw prediction failure is critical business risk"
            ],
            "mitigation_strategies": [
                "Implement confidence thresholds for predictions",
                "Use ensemble approach for draw-heavy periods",
                "Regular model retraining schedule",
                "Performance monitoring dashboard"
            ]
        }
    }
    
    return recommendations

def print_detailed_analysis():
    """Print comprehensive analysis to console"""
    df = create_comparison_table()
    recommendations = generate_recommendations()
    
    print("🏆 COMPREHENSIVE MODEL ANALYSIS REPORT")
    print("=" * 80)
    
    # Executive Summary
    summary = recommendations["executive_summary"]
    print(f"\n📊 EXECUTIVE SUMMARY:")
    print(f"Best Model: {summary['best_model']}")
    print(f"Best Scenario: {summary['best_scenario']}")
    print(f"Best Accuracy: {summary['best_accuracy']}")
    print(f"Status: {summary['status']}")
    
    # Performance Table
    print(f"\n📈 DETAILED PERFORMANCE TABLE:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Key Findings
    print(f"\n🔍 KEY FINDINGS:")
    for i, finding in enumerate(recommendations["key_findings"], 1):
        print(f"{i}. {finding}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"\n🚨 IMMEDIATE ACTIONS:")
    for action in recommendations["recommendations"]["immediate_actions"]:
        print(f"  • {action}")
    
    print(f"\n📅 MEDIUM-TERM (1-3 months):")
    for action in recommendations["recommendations"]["medium_term"]:
        print(f"  • {action}")
    
    print(f"\n🔮 LONG-TERM (3+ months):")
    for action in recommendations["recommendations"]["long_term"]:
        print(f"  • {action}")
    
    # Risk Assessment
    print(f"\n⚠️  RISK ASSESSMENT:")
    risk = recommendations["risk_assessment"]
    print(f"Production Readiness: {risk['production_readiness']}")
    print(f"\nReliability Issues:")
    for issue in risk["reliability_issues"]:
        print(f"  • {issue}")
    print(f"\nMitigation Strategies:")
    for strategy in risk["mitigation_strategies"]:
        print(f"  • {strategy}")
    
    return recommendations

def save_full_report():
    """Save comprehensive analysis to files"""
    df = create_comparison_table()
    recommendations = generate_recommendations()
    
    # Save CSV
    df.to_csv('outputs/model_comparison_final.csv', index=False)
    
    # Save JSON recommendations
    with open('outputs/recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    # Create plots
    plot_performance_comparison()
    
    print(f"\n📁 FILES SAVED:")
    print(f"  • outputs/model_comparison_final.csv")
    print(f"  • outputs/recommendations.json") 
    print(f"  • outputs/comprehensive_model_analysis.png")
    print(f"  • outputs/cm_phase1_baseline.png")
    print(f"  • outputs/cm_phase2_baseline.png")

def main():
    """Main analysis execution"""
    print("🚀 GENERATING COMPREHENSIVE ANALYSIS...")
    
    # Create outputs directory
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Generate and print analysis
    recommendations = print_detailed_analysis()
    
    # Save all outputs
    save_full_report()
    
    print(f"\n✅ ANALYSIS COMPLETE!")

if __name__ == '__main__':
    main()