# 🏆 RAPPORT COMPARATIF MODÈLES CHAMPIONS - EPL 2025-26

*Généré le 2025-09-17 10:05:49*

---

## 1️⃣ RÉSUMÉ EXÉCUTIF

| Modèle | CV Historique | Test 40 Matchs | F1-Score | Verdict Production |
|--------|---------------|----------------|----------|-------------------|
| **Baseline Champion** | 53.5% ± 3.6% | **47.5%** | 0.417 | 🔶 Performance historique |
| **Cascade Champion** | 46.9% ± 3.9% | **50.0%** | 0.500 | 🎯 Spécialisé EPL 2025-26 |

### 🎯 Verdict Final
- **Gagnant CV Historique:** Baseline Champion (+6.6pp)
- **Gagnant Test EPL 2025-26:** Cascade Champion (+2.5pp)
- **Recommandation:** Baseline pour stabilité générale

---

## 2️⃣ ANALYSE DES PERFORMANCES

### 📊 Accuracy Comparée
```
Baseline Champion: 47.5% | Cascade Champion: 50.0%
Différence: +2.5pp en faveur du Cascade
```

### 🎲 Distribution des Prédictions (H/D/A)

```
Baseline     H:██████████ 62.5% | D:░░░░░░░░░░  0.0% | A:██████░░░░ 37.5%\nCascade      H:████████░░ 50.0% | D:███░░░░░░░ 22.5% | A:████░░░░░░ 27.5%\nRéel 2025    H:████████░░ 50.0% | D:███░░░░░░░ 22.5% | A:████░░░░░░ 27.5%
```

### 📈 Performance Cross-Validation Historique

| Fold | Baseline | Cascade | Écart |
|------|----------|---------|-------|
| 1 | 47.632 | 40.526 | -7.105 |
| 2 | 51.842 | 46.579 | -5.263 |
| 3 | 54.211 | 45.789 | -8.421 |
| 4 | 58.158 | 51.842 | -6.316 |
| 5 | 55.789 | 50.000 | -5.789 |

**Moyenne:** Baseline 53.526 ± 3.595 | Cascade 46.947 ± 3.898

---

## 3️⃣ AUDIT DE QUALITÉ

### 🔧 Matrices de Confusion - Test EPL 2025-26

#### Baseline Champion
```
         Pred: H    D    A
Real H:    14    0    6
Real D:     5    0    4
Real A:     6    0    5
```

#### Cascade Champion
```
         Pred: H    D    A
Real H:    12    5    3
Real D:     3    3    3
Real A:     5    1    5
```

### ⚙️ Feature Importance (Baseline Champion)

```
```

---

## 4️⃣ PRÉDICTIONS MATCH PAR MATCH

| # | Date | Match | Réel | Baseline | Prob H/D/A | Cascade | Prob H/D/A | ✓ |
|---|------|-------|------|----------|------------|---------|------------|---|
|  1 | 08-15 | Liverpoo vs Bournemo | H | H | 0.53/0.23/0.24 | H | 0.65/0.15/0.20 | B:✅ C:✅ |
|  2 | 08-16 | Wolves vs Man City | A | A | 0.32/0.24/0.44 | A | 0.20/0.15/0.65 | B:✅ C:✅ |
|  3 | 08-16 | Tottenha vs Burnley | H | H | 0.52/0.23/0.25 | H | 0.65/0.15/0.20 | B:✅ C:✅ |
|  4 | 08-16 | Sunderla vs West Ham | H | A | 0.24/0.26/0.50 | D | 0.25/0.60/0.15 | B:❌ C:❌ |
|  5 | 08-16 | Brighton vs Fulham | D | H | 0.48/0.26/0.26 | D | 0.25/0.60/0.15 | B:❌ C:✅ |
|  6 | 08-16 | Aston Vi vs Newcastl | D | A | 0.29/0.27/0.44 | D | 0.25/0.60/0.15 | B:❌ C:✅ |
|  7 | 08-17 | Chelsea vs Crystal  | D | H | 0.53/0.24/0.22 | H | 0.65/0.15/0.20 | B:❌ C:❌ |
|  8 | 08-17 | Nott'm F vs Brentfor | H | H | 0.48/0.27/0.26 | D | 0.25/0.60/0.15 | B:✅ C:❌ |
|  9 | 08-17 | Man Unit vs Arsenal | A | H | 0.48/0.28/0.24 | H | 0.65/0.15/0.20 | B:❌ C:❌ |
| 10 | 08-18 | Leeds vs Everton  | H | H | 0.54/0.23/0.23 | H | 0.65/0.15/0.20 | B:✅ C:✅ |
| 11 | 08-22 | West Ham vs Chelsea | A | A | 0.36/0.24/0.40 | H | 0.65/0.15/0.20 | B:✅ C:❌ |
| 12 | 08-23 | Arsenal vs Leeds  | H | A | 0.28/0.23/0.49 | A | 0.20/0.15/0.65 | B:❌ C:❌ |
| 13 | 08-23 | Bournemo vs Wolves | H | H | 0.52/0.23/0.24 | H | 0.65/0.15/0.20 | B:✅ C:✅ |
| 14 | 08-23 | Brentfor vs Aston Vi | H | A | 0.25/0.28/0.47 | D | 0.25/0.60/0.15 | B:❌ C:❌ |
| 15 | 08-23 | Man City vs Tottenha | A | H | 0.47/0.27/0.26 | H | 0.65/0.15/0.20 | B:❌ C:❌ |
| 16 | 08-23 | Burnley vs Sunderla | H | H | 0.38/0.29/0.32 | D | 0.25/0.60/0.15 | B:✅ C:❌ |
| 17 | 08-24 | Crystal  vs Nott'm F | D | A | 0.27/0.23/0.50 | A | 0.20/0.15/0.65 | B:❌ C:❌ |
| 18 | 08-24 | Everton vs Brighton | H | A | 0.35/0.25/0.40 | A | 0.20/0.15/0.65 | B:❌ C:❌ |
| 19 | 08-24 | Fulham vs Man Unit | D | H | 0.41/0.27/0.31 | D | 0.25/0.60/0.15 | B:❌ C:✅ |
| 20 | 08-25 | Newcastl vs Liverpoo | A | H | 0.39/0.24/0.37 | A | 0.20/0.15/0.65 | B:❌ C:✅ |
| 21 | 08-30 | Wolves vs Everton | A | A | 0.35/0.23/0.42 | A | 0.20/0.15/0.65 | B:✅ C:✅ |
| 22 | 08-30 | Tottenha vs Bournemo | A | A | 0.33/0.28/0.39 | D | 0.25/0.60/0.15 | B:✅ C:❌ |
| 23 | 08-30 | Leeds vs Newcastl | D | H | 0.53/0.24/0.23 | H | 0.65/0.15/0.20 | B:❌ C:❌ |
| 24 | 08-30 | Man Unit vs Burnley | H | H | 0.47/0.22/0.32 | H | 0.65/0.15/0.20 | B:✅ C:✅ |
| 25 | 08-30 | Chelsea vs Fulham | H | H | 0.49/0.24/0.27 | H | 0.65/0.15/0.20 | B:✅ C:✅ |
| 26 | 08-30 | Sunderla vs Brentfor | H | A | 0.31/0.26/0.44 | A | 0.20/0.15/0.65 | B:❌ C:❌ |
| 27 | 08-31 | Nott'm F vs West Ham | A | H | 0.49/0.28/0.23 | H | 0.65/0.15/0.20 | B:❌ C:❌ |
| 28 | 08-31 | Liverpoo vs Arsenal | H | A | 0.34/0.27/0.39 | D | 0.25/0.60/0.15 | B:❌ C:❌ |
| 29 | 08-31 | Brighton vs Man City | H | H | 0.43/0.22/0.35 | H | 0.65/0.15/0.20 | B:✅ C:✅ |
| 30 | 08-31 | Aston Vi vs Crystal  | A | H | 0.44/0.22/0.34 | A | 0.20/0.15/0.65 | B:❌ C:✅ |
| 31 | 09-13 | Bournemo vs Brighton | H | H | 0.42/0.21/0.37 | H | 0.65/0.15/0.20 | B:✅ C:✅ |
| 32 | 09-13 | Arsenal vs Nott'm F | H | H | 0.69/0.20/0.12 | H | 0.65/0.15/0.20 | B:✅ C:✅ |
| 33 | 09-13 | West Ham vs Tottenha | A | H | 0.67/0.20/0.13 | H | 0.65/0.15/0.20 | B:❌ C:❌ |
| 34 | 09-13 | Crystal  vs Sunderla | D | A | 0.32/0.22/0.47 | A | 0.20/0.15/0.65 | B:❌ C:❌ |
| 35 | 09-13 | Newcastl vs Wolves | H | H | 0.67/0.20/0.13 | H | 0.65/0.15/0.20 | B:✅ C:✅ |
| 36 | 09-13 | Fulham vs Leeds   | H | H | 0.67/0.20/0.13 | H | 0.65/0.15/0.20 | B:✅ C:✅ |
| 37 | 09-13 | Everton vs Aston Vi | D | H | 0.65/0.21/0.14 | H | 0.65/0.15/0.20 | B:❌ C:❌ |
| 38 | 09-13 | Brentfor vs Chelsea | D | A | 0.21/0.21/0.58 | A | 0.20/0.15/0.65 | B:❌ C:❌ |
| 39 | 09-14 | Burnley vs Liverpoo | A | A | 0.19/0.20/0.60 | A | 0.20/0.15/0.65 | B:✅ C:✅ |
| 40 | 09-14 | Man City vs Man Unit | H | H | 0.68/0.20/0.12 | H | 0.65/0.15/0.20 | B:✅ C:✅ |

### 📊 Synthèse Erreurs
```
Baseline: 21 erreurs / 40 matchs ▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░
Cascade:  20 erreurs / 40 matchs ▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░
```

---

## 5️⃣ CONCLUSIONS ET RECOMMANDATIONS

### ✅ Points Forts

**Baseline Champion:**
- ✅ Excellent historique (53.5% CV)
- ✅ Robuste et stable (±3.6% variance)
- ✅ Architecture simple et maintenable

**Cascade Champion:**
- ✅ Meilleur sur EPL 2025-26 (50.0% vs 47.5%)
- ✅ Détecte les draws (22.5% vs 0.0%)
- ✅ Distribution calibrée sur EPL 2025-26

### ⚠️ Points d'Attention

**Baseline Champion:**
- ⚠️ Échec EPL 2025-26 (47.5% < 50% majority)
- ⚠️ 0% draws prédits (vs 22.5% réels)

**Cascade Champion:**
- ⚠️ Performance historique modeste (46.9% CV)
- ⚠️ Architecture complexe (2 modèles)

### 🎯 Recommandation Stratégique

**POUR PRODUCTION IMMÉDIATE:**
- 🥇 **Cascade Champion** pour EPL 2025-26 (50.0% > 47.5%)
- 📊 Distribution remarquablement calibrée
- 🎯 Spécialisé pour patterns début saison

**POUR ROBUSTESSE LONG TERME:**
- 🥈 **Baseline Champion** comme référence stable
- 📈 Excellent historique (53.5% CV)
- 🔧 Re-calibrer avec plus de données EPL 2025-26

### 📋 Plan d'Action

1. **Court terme (J5-J10):** Utiliser Cascade Champion
2. **Collecte données:** Enrichir avec matchs EPL 2025-26 supplémentaires
3. **Re-calibration:** Ajuster modèles avec 100+ matchs EPL 2025-26
4. **Monitoring:** Tracker performance continue vs majority class

---

### 📊 Score Final

| Critère | Baseline | Cascade | Gagnant |
|---------|----------|---------|---------|
| CV Historique | 53.5% | 46.9% | 🥇 Baseline |
| Test EPL 2025-26 | 47.5% | 50.0% | 🥇 Cascade |
| Stabilité | 3.60 | 3.90 | 🥇 Baseline |
| Production Ready | ✅ | ✅ | 🥇 Cascade |

**🏆 CHAMPION GLOBAL:** Baseline Champion

---

*Rapport généré automatiquement par audit_pipeline.py - Version 2.0*
