#!/usr/bin/env python3
"""
🔬 SCIENTIFIC DASHBOARD
=======================
Dashboard technique pour validation modèles et métriques approfondies.
Target: Data scientists, équipe technique.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_loader import calculate_performance_metrics, load_metadata

def show_scientific_dashboard():
    """Interface principale Scientific Dashboard."""
    
    st.header("🔬 Scientific Dashboard")
    st.markdown("**Validation technique et métriques approfondies des modèles**")
    st.markdown("---")
    
    # Métriques essentielles
    show_core_metrics()
    
    st.markdown("---")
    
    # Placeholder sections futures
    show_future_features()

def show_core_metrics():
    """Affiche les métriques de validation essentielles."""
    
    st.subheader("📊 Métriques de Validation")
    
    metrics = calculate_performance_metrics()
    
    if not metrics:
        st.error("❌ Métriques non disponibles")
        return
    
    # Performance principale
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cascade_test = metrics['cascade']['test_accuracy']
        st.metric("🎯 Cascade Test", f"{cascade_test:.1f}%", "EPL 2025-26")
    
    with col2:
        cascade_cv = metrics['cascade']['cv_accuracy']  
        cascade_std = metrics['cascade']['cv_std']
        st.metric("🎯 Cascade CV", f"{cascade_cv:.1f}%", f"±{cascade_std:.1f}%")
    
    with col3:
        baseline_test = metrics['baseline']['test_accuracy']
        st.metric("⚡ Baseline Test", f"{baseline_test:.1f}%", "EPL 2025-26") 
    
    with col4:
        baseline_cv = metrics['baseline']['cv_accuracy']
        baseline_std = metrics['baseline']['cv_std']
        st.metric("⚡ Baseline CV", f"{baseline_cv:.1f}%", f"±{baseline_std:.1f}%")
    
    # Stabilité modèles
    st.markdown("#### 🎯 Analyse Stabilité")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cascade_stability = metrics['cascade']['stability']
        st.info(f"**Cascade Stability**: {cascade_stability}")
        st.caption("Variance cross-validation")
    
    with col2:
        baseline_stability = metrics['baseline']['stability']  
        st.info(f"**Baseline Stability**: {baseline_stability}")
        st.caption("Variance cross-validation")
    
    # Comparaison baselines
    st.markdown("#### 📈 Performance vs Targets")
    
    baselines_table = pd.DataFrame({
        'Baseline': ['Random (33.3%)', 'Always Home (43.6%)', 'Good Target (50%)', 'Excellent Target (55%)'],
        'Cascade Performance': [
            f"+{cascade_test - 33.3:.1f}pp",
            f"+{cascade_test - 43.6:.1f}pp", 
            f"+{cascade_test - 50.0:.1f}pp",
            f"+{cascade_test - 55.0:.1f}pp"
        ],
        'Baseline Performance': [
            f"+{baseline_test - 33.3:.1f}pp",
            f"+{baseline_test - 43.6:.1f}pp",
            f"+{baseline_test - 50.0:.1f}pp", 
            f"+{baseline_test - 55.0:.1f}pp"
        ],
        'Cascade Status': [
            "✅ Beaten" if cascade_test > 33.3 else "❌ Failed",
            "✅ Beaten" if cascade_test > 43.6 else "❌ Failed", 
            "✅ Beaten" if cascade_test > 50.0 else "❌ Failed",
            "✅ Beaten" if cascade_test > 55.0 else "❌ Failed"
        ],
        'Baseline Status': [
            "✅ Beaten" if baseline_test > 33.3 else "❌ Failed",
            "✅ Beaten" if baseline_test > 43.6 else "❌ Failed",
            "✅ Beaten" if baseline_test > 50.0 else "❌ Failed", 
            "✅ Beaten" if baseline_test > 55.0 else "❌ Failed"
        ]
    })
    
    st.dataframe(baselines_table, width='stretch')

def show_future_features():
    """Aperçu fonctionnalités futures."""
    
    st.subheader("🚧 Fonctionnalités en Développement")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Diagnostics", "📊 Visualisations", "📋 Audit"])
    
    with tab1:
        st.markdown("""
        #### 🔬 Diagnostics Avancés
        - **Feature Drift Detection**: Comparaison distributions 2019-2024 vs 2025-26
        - **Concept Drift Monitoring**: Alerte dégradation performance
        - **Calibration Analysis**: Probabilités prédites vs réelles
        - **Robustness Testing**: Tests multi-seeds stabilité
        """)
    
    with tab2:
        st.markdown("""
        #### 📈 Visualisations Techniques  
        - **Confusion Matrix Interactive**: H/D/A détaillée pour chaque modèle
        - **Feature Importance**: Top 10 features avec comparaison
        - **ROC Curves**: Analyse performance par classe
        - **Learning Curves**: Évolution performance vs données
        """)
    
    with tab3:
        st.markdown("""
        #### 🛡️ Pipeline d'Audit
        - **Data Quality Checks**: Validation intégrité données
        - **Model Validation**: Tests reproductibilité 
        - **Performance Regression**: Alerte dégradation
        - **Compliance Reporting**: Rapports audit automatiques
        """)
    
    # Mock confusion matrix preview
    st.markdown("---")
    st.subheader("👀 Aperçu: Confusion Matrix")
    
    # Simulation données confusion matrix
    confusion_data = {
        'Predicted': ['Home', 'Home', 'Home', 'Draw', 'Draw', 'Draw', 'Away', 'Away', 'Away'],
        'Actual': ['Home', 'Draw', 'Away', 'Home', 'Draw', 'Away', 'Home', 'Draw', 'Away'],
        'Cascade': [12, 5, 3, 3, 3, 3, 5, 1, 5],
        'Baseline': [14, 0, 6, 5, 0, 4, 6, 0, 5] 
    }
    
    confusion_df = pd.DataFrame(confusion_data)
    st.dataframe(confusion_df, width='stretch')
    st.caption("Données EPL 2025-26 - 40 matchs test")

if __name__ == "__main__":
    st.set_page_config(page_title="Scientific Dashboard Test", layout="wide")
    show_scientific_dashboard()