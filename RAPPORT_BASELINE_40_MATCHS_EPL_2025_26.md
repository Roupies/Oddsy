# 🏆 RAPPORT BASELINE 40 MATCHS EPL 2025-26

**Date du rapport:** 16/09/2025 11:47

## 📊 Performance Globale

- **Accuracy:** 52.5% (21/40 matchs corrects)
- **Modèle:** Cascade Draw vs NotDraw → Home vs Away
- **Features:** 10 features v2.3 validées
- **Entrainement:** 2,280 matchs historiques (< 2025-08-01)
- **Test:** 40 matchs EPL 2025-26 J1-J4

## 🎯 Distribution des Résultats

### Réalité EPL 2025-26

- **Home:** 50.0% (20 matchs)
- **Draw:** 22.5% (9 matchs)
- **Away:** 27.5% (11 matchs)

### Prédictions Cascade
- **Home:** 57.5% (23 prédictions)
- **Draw:** 0.0% (0 prédictions)
- **Away:** 42.5% (17 prédictions)

## 📈 Matrice de Confusion

```
           Prédictions
Réalité    H    D    A    Total   Precision
   H    14    0    6    20     70.0%
   D     5    0    4     9     0.0%
   A     4    0    7    11     63.6%

```

## 📈 Comparaison Benchmarks

- **vs Random (33.3%):** +19.2%
- **vs Majority Class (50.0%):** +2.5%
- **Niveau:** ✅ BON

## 🎯 Détail des 40 Matchs

| # | Équipe Domicile | Équipe Extérieur | Réel | Prédit | Correct |
|---|----------------|-----------------|------|--------|---------|
|  1 | Liverpool | Bournemouth | H | H | ✅ |
|  2 | Aston Villa | Newcastle | D | A | ❌ |
|  3 | Brighton | Fulham | D | H | ❌ |
|  4 | Sunderland | West Ham | H | A | ❌ |
|  5 | Wolves | Man City | A | A | ✅ |
|  6 | Tottenham | Burnley | H | H | ✅ |
|  7 | Chelsea | Crystal Palace | D | H | ❌ |
|  8 | Nott'm Forest | Brentford | H | H | ✅ |
|  9 | Man United | Arsenal | A | H | ❌ |
| 10 | Leeds | Everton | H | H | ✅ |
| 11 | West Ham | Chelsea | A | A | ✅ |
| 12 | Bournemouth | Wolves | H | H | ✅ |
| 13 | Man City | Tottenham | A | H | ❌ |
| 14 | Burnley | Sunderland | H | H | ✅ |
| 15 | Arsenal | Leeds | H | A | ❌ |
| 16 | Brentford | Aston Villa | H | A | ❌ |
| 17 | Crystal Palace | Nott'm Forest | D | A | ❌ |
| 18 | Everton | Brighton | H | A | ❌ |
| 19 | Fulham | Man United | D | H | ❌ |
| 20 | Newcastle | Liverpool | A | A | ✅ |
| 21 | Sunderland | Brentford | H | A | ❌ |
| 22 | Chelsea | Fulham | H | H | ✅ |
| 23 | Man United | Burnley | H | H | ✅ |
| 24 | Tottenham | Bournemouth | A | A | ✅ |
| 25 | Wolves | Everton | A | A | ✅ |
| 26 | Leeds | Newcastle | D | H | ❌ |
| 27 | Brighton | Man City | H | H | ✅ |
| 28 | Nott'm Forest | West Ham | A | H | ❌ |
| 29 | Liverpool | Arsenal | H | H | ✅ |
| 30 | Aston Villa | Crystal Palace | A | A | ✅ |
| 31 | Arsenal | Nott'm Forest | H | H | ✅ |
| 32 | Newcastle | Wolves | H | H | ✅ |
| 33 | Fulham | Leeds | H | H | ✅ |
| 34 | Bournemouth | Brighton | H | A | ❌ |
| 35 | Crystal Palace | Sunderland | D | A | ❌ |
| 36 | Everton | Aston Villa | D | H | ❌ |
| 37 | West Ham | Tottenham | A | H | ❌ |
| 38 | Brentford | Chelsea | D | A | ❌ |
| 39 | Burnley | Liverpool | A | A | ✅ |
| 40 | Man City | Man United | H | H | ✅ |


## 🔍 Analyse par Type de Match

### Matchs à Domicile (H)
- **Performance:** 70.0% (14/20)

### Matchs Nuls (D)
- **Performance:** 0.0% (0/9)

### Matchs à l'Extérieur (A)
- **Performance:** 63.6% (7/11)

## ⚠️ Observations

### Points Forts
- Bonne performance globale (52.5%)
- Anti-leakage temporel strict respecté
- 2,280 matchs d'entrainement robustes

### Points d'Attention
- ⚠️ **Aucun match nul prédit** - Limitation du modèle cascade

## 🎯 Conclusion

La baseline Oddsy sur 40 matchs EPL 2025-26 J1-J4 atteint **52.5% de précision** avec le modèle cascade.

**Verdict:** BON - Acceptable pour production

---
*Rapport généré automatiquement par Oddsy v2.3 - 16/09/2025 11:47*
