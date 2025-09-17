#!/usr/bin/env python3
"""
🎓 EDUCATIONAL DASHBOARD
========================
Dashboard éducatif pour comprendre le fonctionnement des modèles.
Focus: Explications simples, analyses match par match, glossaire.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_loader import get_epl_2025_26_matches, load_metadata

def show_educational_dashboard():
    """Interface principale Educational Dashboard."""
    
    st.header("🎓 Educational Dashboard")
    st.markdown("**Understanding how AI predicts Premier League matches**")
    st.markdown("---")
    
    # Placeholder pour développement futur
    st.info("🚧 **Under Development**")
    
    st.markdown("""
    ### 🎯 Fonctionnalités Prévues:
    
    #### 📊 Match Analyzer
    - Sélection d'un match EPL 2025-26
    - Breakdown des features qui ont influencé la prédiction
    - Comparaison visuelle des équipes
    
    #### 🧠 Comment ça Marche?
    - Explication interactive des 10 features
    - Calculateur Elo rating en temps réel
    - Démystification "market entropy" et "xG efficiency"
    
    #### 🏗️ Architecture Cascade Expliquée
    - Visualisation 2-étapes: Draw Detection → H/A Classification
    - Animation flow de décision
    - Comparaison avec Baseline (single-stage)
    
    #### 📚 Glossaire Interactif
    - Définitions techniques simplifiées
    - Exemples concrets pour chaque concept
    - Quiz validation compréhension
    """)
    
    # Preview match analyzer
    st.markdown("---")
    st.subheader("🔍 Aperçu: Match Analyzer")
    
    epl_matches = get_epl_2025_26_matches()
    if not epl_matches.empty:
        match_options = epl_matches['Match'].tolist()[:10]  # Premier 10 pour demo
        
        selected_match = st.selectbox(
            "Sélectionner un match à analyser:",
            options=match_options,
            help="Choisissez un match pour voir le breakdown de la prédiction"
        )
        
        if selected_match:
            st.success(f"✅ Match sélectionné: **{selected_match}**")
            st.info("🔜 Analyse détaillée disponible dans la version complète")
    
    else:
        st.warning("⚠️ Données matches non disponibles")

if __name__ == "__main__":
    st.set_page_config(page_title="Educational Dashboard Test", layout="wide")
    show_educational_dashboard()