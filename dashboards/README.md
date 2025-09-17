# 🏆 Oddsy Dashboard - Prototype Streamlit

Prototype interactif pour les prédictions Premier League avec architecture dual champions.

## 🚀 Quick Start

### Installation
```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run streamlit_app.py
```

### Accès
- **URL locale**: http://localhost:8501
- **Navigation**: Sidebar pour changer de dashboard

## 📊 Dashboards Disponibles

### 📈 Commercial Dashboard (Priorité 1)
**Target**: Stakeholders business, décideurs
**Approche**: "Prouve-le, puis Utilise-le"

**Section Crédibilité:**
- Performance validée EPL 2025-26 (40 matchs)
- KPIs: Cascade 50.0%, Baseline 47.5%
- Comparaison vs baselines naïves
- Trend performance par journée

**Section Action:**
- Prédictions prochains matchs (J5)
- Recommandation modèle optimal
- Scores de confiance
- Logique Cascade (early) vs Baseline (established)

### 🎓 Educational Dashboard (Développement)
**Target**: Équipe élargie, non-techniques
**Focus**: Explications, analyses match par match

**Fonctionnalités prévues:**
- Match Analyzer interactif
- Explication features (Elo, xG, entropy)  
- Architecture Cascade visualisée
- Glossaire technique simplifié

### 🔬 Scientific Dashboard (Version Lite)
**Target**: Data scientists, validation technique
**Focus**: Métriques, diagnostics

**Disponible:**
- Métriques core (CV, Test accuracy)
- Comparaisons baselines
- Analyse stabilité modèles
- Preview confusion matrix

## 🏗️ Architecture Technique

```
dashboards/
├── streamlit_app.py              # Point d'entrée principal
├── core/                         # Modules partagés
│   ├── data_loader.py           # Cache Streamlit optimisé
│   └── __init__.py
├── pages/                        # Dashboards
│   ├── commercial.py            # Business dashboard
│   ├── educational.py           # Explications
│   ├── scientific.py            # Validation technique
│   └── __init__.py
├── requirements.txt             # Dependencies
└── README.md                    # Ce fichier
```

## 🔧 Performance & Cache

**Stratégie Cache:**
- `@st.cache_data(ttl=3600)` pour données statiques (1h)
- `@st.cache_resource` pour modèles lourds (permanent)
- `@st.cache_data(ttl=1800)` pour prédictions "live" (30min)

**Optimisations:**
- Chargement unique dataset (2,320 matches)
- Metadata JSON centralisé
- Simulation prédictions futures (pas de recalcul temps réel)

## 📈 Données Sources

**Dataset Principal:**
- `data/processed/v_auto_update_20250916_110247.csv`
- 2,320 matches (2019-2026)
- EPL 2025-26: 40 matches validés

**Métadonnées Modèles:**
- `models/production/baseline_champion_v23_metadata.json`
- `models/production/cascade_champion_v2_metadata.json`
- Performance, features, audit complets

**Modèles (Locaux uniquement):**
- `models/production/baseline_champion_v23.joblib` (55MB)
- `models/production/cascade_champion_v2.joblib` (11MB)

## 🎯 Plan Développement

### ✅ Phase 1 Complète (Jours 1-2)
- [x] Structure multi-pages avec sidebar
- [x] Core data_loader avec cache optimisé
- [x] Commercial Dashboard avec "Prouve + Utilise" 
- [x] Educational/Scientific stubs fonctionnels

### 🔄 Phase 2 (Jours 3-5) 
- [ ] Intégration modèle Baseline réel
- [ ] Reconstruction Cascade Champion
- [ ] Prédictions futures algorithme
- [ ] Performance optimization

### 🚀 Phase 3 (Jours 6-10)
- [ ] Educational Dashboard complet
- [ ] Scientific Dashboard diagnostics
- [ ] UX/UI polish
- [ ] Tests & documentation

## 🎭 Simulation vs Réel

**Actuellement Simulé:**
- Prédictions futures (matchs J5+)
- Reconstruction Cascade Champion
- Trends performance par journée

**Données Réelles:**
- Métriques performance (metadata JSON)
- 40 matchs EPL 2025-26 validés
- Features et architecture modèles

## 🚨 Troubleshooting

**Erreur "ModuleNotFoundError: streamlit":**
```bash
pip install streamlit plotly pandas numpy scikit-learn joblib
```

**Erreur "Données non disponibles":**
- Vérifier paths vers `data/processed/` et `models/production/`
- S'assurer que le script est lancé depuis la racine du projet

**Performance lente:**
- Clear cache Streamlit: `streamlit cache clear`
- Relancer application: `Ctrl+C` puis `streamlit run streamlit_app.py`

## 📞 Support

Prototype développé pour validation concept "Prouve-le, puis Utilise-le".
Focus maximum impact business en 2 semaines.