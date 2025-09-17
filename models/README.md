# 🤖 Modèles Oddsy

Organisation des modèles de machine learning pour la prédiction EPL.

## 📁 Structure

### `/production/` - Modèles de Production
- `baseline_champion_v23.joblib` - **Baseline Champion** (RandomForest 53.5% CV, 47.5% EPL 2025-26)
- `cascade_champion_v2.joblib` - **Cascade Champion** (Architecture 2-étapes, 50.0% EPL 2025-26)
- `*_metadata.json` - Métadonnées complètes des modèles (features, hyperparamètres, performance)

### `/experimental/` - Modèles Expérimentaux
Modèles en cours de développement ou archivés après validation.

## 🏆 Modèles Champions Actuels

### Baseline Champion v2.3
- **Architecture:** RandomForest + CalibratedClassifierCV
- **Features:** 10 features optimisées (elo_diff, market_entropy, xG_efficiency, etc.)
- **Performance CV:** 53.5% ± 3.6% (historique 2019-2025)
- **Performance EPL 2025-26:** 47.5% (40 matchs test)
- **Statut:** ✅ Production ready - Stable long terme

### Cascade Champion v2.0
- **Architecture:** Binary (Draw/Non-Draw) → Ternary (H/A)
- **Innovation:** Spécialisation détection draws
- **Performance EPL 2025-26:** 50.0% (40 matchs test)
- **Performance CV:** 46.9% ± 3.9% (historique)
- **Statut:** ✅ Production ready - Optimisé early-season

## 🎯 Utilisation

### Chargement Modèles
```python
import joblib

# Baseline Champion
baseline = joblib.load('models/production/baseline_champion_v23.joblib')

# Cascade Champion
cascade = joblib.load('models/production/cascade_champion_v2.joblib')
```

### Prédictions
```python
# Features standardisées (10 features)
X_new = prepare_features(match_data)

# Prédictions
baseline_pred = baseline.predict(X_new)
cascade_pred = cascade.predict(X_new)
```

## 📊 Benchmarks

| Modèle | CV Historique | EPL 2025-26 | Use Case |
|--------|---------------|-------------|----------|
| Baseline Champion | 53.5% | 47.5% | Stabilité long terme |
| Cascade Champion | 46.9% | 50.0% | Early-season, détection draws |

---

*Mise à jour: 2025-09-17 - Post analyse 2 champions*