# CLAUDE.md

## Projet Oddsy - Prédiction Football EPL

**Objectif:** Modèles ML pour prédire les résultats de matchs Premier League (H/D/A)

## 🎯 Règles de Développement

### Performance Targets
- **Minimum:** > 43.6% (baseline majoritaire)
- **Objectif:** > 50% accuracy
- **Excellence:** > 55%

### Méthodologie Obligatoire
- **Validation temporelle stricte** (TimeSeriesSplit)
- **Anti-data leakage** (features calculées AVANT match_date)
- **Test final sur données unseen** (EPL 2025-26)

### Standards de Code
- Toujours utiliser `verbose=2` pour optimisations longues
- Préférer `market_entropy_norm` à `market_entropy_historical`
- Sauvegarder modèles avec metadata complète
- Tests obligatoires avant production

### Architecture Actuelle
- **Production:** Dual Champions (Baseline + Cascade)
- **Dataset principal:** `data/processed/v_auto_update_*.csv`
- **Modèles:** `models/production/`

### Commandes Essentielles
```bash
# Tests rapides
python scripts/analysis/test_models.py

# Optimisation complète  
jupyter notebook Optimization_Notebook.ipynb

# Production dashboard
python dashboards/main.py
```

## 📁 Structure Projet
- `database/` - PostgreSQL setup + connecteur Python
- `data/` - Datasets versionnés
- `models/production/` - Modèles validés
- `scripts/` - Scripts par catégorie
- `experiments/` - Code expérimental
- `predictions/` - Fichiers JSON de prédictions
- `docs/` - Documentation et rapports
- `temp/` - Fichiers temporaires

## 🗄️ Base de Données
```bash
# Démarrer PostgreSQL
docker-compose up -d

# Se connecter
docker exec -it oddsy_postgres psql -U oddsy_user -d oddsy_football

# Admin interface: http://localhost:8080
```

## 🏆 Production Status
- **Baseline Champion v2.3:** 53.5% CV stable
- **Cascade Champion v2.0:** 50.0% EPL 2025-26 (spécialiste draws)
- **Status:** Production ready avec dual deployment