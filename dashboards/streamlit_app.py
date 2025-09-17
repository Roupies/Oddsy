#!/usr/bin/env python3
"""
🏆 ODDSY DASHBOARD - MAIN APPLICATION
====================================
Prototype Streamlit pour prédictions Premier League
Architecture: Commercial-First avec approche "Prouve-le, puis Utilise-le"

Audiences:
- 📈 Commercial: Stakeholders business (Prédictions + Performance)
- 🎓 Educational: Équipe élargie (Explications + Analyses)
- 🔬 Scientific: Data scientists (Validation + Métriques)
"""

import streamlit as st
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dashboards.pages import commercial, educational, scientific

# Page configuration
st.set_page_config(
    page_title="Oddsy - Premier League Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Oddsy - AI-powered Premier League match predictions with dual champions architecture"
    }
)

def main():
    """Application principale avec navigation sidebar."""
    
    # Header global
    st.title("⚽ Oddsy - Premier League AI Predictions")
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        st.markdown("Choose your view based on your profile:")
        
        page = st.selectbox(
            "Select Dashboard",
            options=[
                "📈 Commercial - Business View",
                "🎓 Educational - Technical Insights", 
                "🔬 Scientific - Model Validation"
            ],
            index=0,  # Default to commercial
            help="Commercial: Predictions & Performance | Educational: Explanations & Analysis | Scientific: Validation & Metrics"
        )
        
        st.markdown("---")
        
        # Project info
        with st.expander("ℹ️ About Oddsy"):
            st.markdown("""
            **Oddsy** uses artificial intelligence to predict Premier League results.
            
            **🏆 Dual Champions Architecture:**
            - **Baseline Champion:** 53.5% accuracy (long-term stability)
            - **Cascade Champion:** 50.0% EPL 2025-26 (draw detection)
            
            **📊 Validated Performance:**
            - 40 EPL 2025-26 matches tested
            - Beats naive baselines (+6.4pp vs Always Home)
            - Rigorous temporal validation
            """)
        
        # Model status
        with st.expander("🤖 Model Status"):
            st.success("✅ Baseline Champion: Production Ready")
            st.success("✅ Cascade Champion: Production Ready") 
            st.info("📊 Dataset: 2,320 matches (2019-2026)")
            st.info("🎯 Features: 10 optimized (Elo, xG, Market)")
    
    # Route to selected page
    if "Commercial" in page:
        commercial.show_commercial_dashboard()
    elif "Educational" in page:
        educational.show_educational_dashboard()
    elif "Scientific" in page:
        scientific.show_scientific_dashboard()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 0.8em;'>
            🏆 Oddsy v2.0 - Dual Champions Architecture<br>
            Baseline (53.5% CV) + Cascade (50.0% EPL 2025-26)
            </div>
            """, 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()