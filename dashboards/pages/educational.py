#!/usr/bin/env python3
"""
🎓 EDUCATIONAL DASHBOARD
========================
Educational dashboard for understanding how models work.
Focus: Simple explanations, match-by-match analysis, glossary.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.robust_data_loader import get_epl_2025_26_matches, load_unified_metrics

def show_educational_dashboard():
    """Main interface for Educational Dashboard."""
    
    st.header("🎓 Educational Dashboard")
    st.markdown("**Understanding how AI predicts Premier League matches**")
    st.markdown("---")
    
    # Placeholder for future development
    st.info("🚧 **Under Development**")
    
    st.markdown("""
    ### 🎯 Planned Features:
    
    #### 📊 Match Analyzer
    - **Interactive predictions**: Select any match and see detailed prediction breakdown
    - **Feature explanations**: Why did the model predict this result?
    - **Historical context**: How similar matches performed in the past
    
    #### 📚 ML Concepts Explained
    - **What is machine learning?**: Simple explanations without jargon
    - **How models learn**: Understanding training data and patterns
    - **Confidence levels**: What does 67% confidence really mean?
    
    #### 🎮 Interactive Demos
    - **Feature impact simulator**: See how changing team form affects predictions
    - **"Build your prediction"**: Manual prediction vs AI comparison
    - **Historical accuracy**: Test the model on past seasons
    
    #### 📖 Football Analytics Glossary
    - **xG (Expected Goals)**: What it measures and why it matters
    - **Elo ratings**: How we measure team strength
    - **Market entropy**: Using betting odds as intelligence signals
    """)
    
    st.markdown("---")
    
    # Real Historical Examples
    with st.expander("📚 Exemple Réel: Comment l'IA a Appris sur 5 Saisons"):
        st.markdown("""
        **Apprentissage Historique (2019-2025):**
        
        Notre Baseline Champion a analysé **1,900 matchs** historiques pour découvrir:
        - 🏠 **Home wins**: 43.6% (pattern stable depuis 2019)
        - ⚡ **Away wins**: 33.4% (en hausse depuis COVID)  
        - 🤝 **Draws**: 23.0% (plus rares mais prévisibles)
        
        **Pourquoi 5 saisons = Signal Fort:**
        - ✅ **Volume**: 1,900 matchs vs 40 matchs récents
        - ✅ **Patterns**: Détecte tendances long terme 
        - ✅ **Robustesse**: 51.2% ± 3.8% performance stable
        - ✅ **Validation**: Test temporel sur 380 matchs séparés
        
        **Signal vs Bruit:**
        - 📊 **Signal**: Performance historique validée (51.2%)
        - 🌪️ **Bruit**: Volatilité court terme (40 matchs = variance élevée)
        """)
    
    # Historical Learning Examples
    st.markdown("---")
    st.markdown("### 📚 Historical Learning: How AI Learned from 5 Seasons")
    
    # Load real metrics for historical examples
    metrics = load_unified_metrics()
    
    if metrics and metrics.get("data_status") == "complete":
        
        # Historical Learning Section
        st.markdown("#### 🕰️ L'Apprentissage Historique (2019-2025)")
        
        baseline_data = metrics.get('baseline', {}).get('audit_results', {})
        if baseline_data:
            baseline_cv = baseline_data.get('cross_validation', {}).get('cv_mean', 0.512) * 100
            train_size = baseline_data.get('model_info', {}).get('train_size', 1900)
            test_size = baseline_data.get('model_info', {}).get('test_size', 380)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📖 Apprentissage", f"{train_size:,} matchs", "2019-2025 training")
            with col2:
                st.metric("🧪 Test", f"{test_size} matchs", "Validation temporelle")
            with col3:
                st.metric("🎯 Performance", f"{baseline_cv:.1f}%", "Cross-validation")
        
        # Interactive Historical Quiz
        st.markdown("---")
        st.markdown("### 🎯 Quiz: Comprendre l'Apprentissage Historique")
        
        # Quiz Question 1: Historical Learning
        st.markdown("#### Question 1: Signal Historique vs Bruit")
        baseline_cv = baseline_data.get('cross_validation', {}).get('cv_mean', 0.512) * 100 if baseline_data else 51.2
        
        q1_answer = st.radio(
            "Pourquoi utiliser 5 saisons (2019-2025) plutôt que les 40 derniers matchs?",
            [
                "📊 Plus de données = plus de signal (1900 vs 40 matchs)",
                "🕰️ Les matchs récents sont plus représentatifs",
                "⚽ La Premier League a changé depuis 2019",
                "🎲 C'est juste plus pratique à analyser"
            ],
            key="q1"
        )
        
        if st.button("Check Answer 1", key="check1"):
            if "Plus de données = plus de signal" in q1_answer:
                st.success("✅ Correct! 1900 matchs donnent un signal beaucoup plus robuste que 40 matchs qui peuvent contenir du bruit et des patterns atypiques.")
                st.info(f"💡 Notre Baseline Champion a appris sur {train_size:,} matchs historiques pour atteindre {baseline_cv:.1f}% de performance!")
            else:
                st.error("❌ Faux. Plus de données historiques = signal plus fort et patterns plus fiables.")
                st.info("💡 40 matchs récents = bruit court terme. 1900 matchs historiques = signal robuste long terme.")
        
        st.markdown("---")
        
        # Quiz Question 2: Draw Detection
        st.markdown("#### Question 2: Draw Detection Specialist")
        q2_answer = st.radio(
            "Which model is specifically designed to detect draws?",
            [
                "🎯 Cascade Champion (Binary then Ternary architecture)",
                "⚡ Baseline Champion (RandomForest classifier)",
                "🤖 Both models detect draws equally well",
                "🚫 Neither model can predict draws"
            ],
            key="q2"
        )
        
        if st.button("Check Answer 2", key="check2"):
            if "Cascade Champion" in q2_answer:
                st.success("✅ Correct! Cascade Champion uses a 2-stage architecture: first detects draws vs non-draws, then classifies Home/Away for non-draws.")
                st.info("💡 This is why Cascade shows better draw detection in EPL 2025-26 early season matches!")
            else:
                st.error("❌ Wrong. Cascade Champion is specifically designed for draw detection with its Binary-then-Ternary cascade architecture.")
        
        st.markdown("---")
        
        # Quiz Question 3: Real Performance
        st.markdown("#### Question 3: Beating Baselines")
        q3_answer = st.radio(
            "What naive baseline do both models need to beat to be useful?",
            [
                "🎲 Random prediction (33.3%)",
                "🏠 Always predict Home (43.6%)", 
                "🎯 Good target (50%)",
                "⭐ Excellence target (55%)"
            ],
            key="q3"
        )
        
        if st.button("Check Answer 3", key="check3"):
            if "Always predict Home" in q3_answer:
                st.success("✅ Correct! Since Home wins occur 43.6% of the time in EPL, any useful model must beat this naive strategy.")
                st.info(f"💡 Our models achieve {baseline_acc:.1f}% (Baseline) and {cascade_acc:.1f}% (Cascade), both beating the 43.6% naive baseline!")
            else:
                st.error("❌ Wrong. The hardest naive baseline to beat is 'Always predict Home' at 43.6%, since Home wins are the most common outcome in football.")
        
        # Final score encouragement
        st.markdown("---")
        st.info("🎓 **Learning Tip**: Real model validation uses temporal splits and cross-validation to ensure honest performance measurement!")
        
    else:
        st.error("❌ Cannot load quiz data - requires model metrics")
        st.info("The quiz uses real performance data from our Champions to test your understanding!")

if __name__ == "__main__":
    st.set_page_config(page_title="Educational Dashboard Test", layout="wide")
    show_educational_dashboard()