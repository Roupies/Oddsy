# 📊 VRAIES FEATURES J1 EPL 2025-26 - MODÈLE CASCADE ÉQUILIBRÉ

**Date du rapport:** 16/09/2025 15:12

## 🎯 Contexte et Objectif

Ce rapport présente les **vraies features d'entrée** utilisées par le modèle cascade équilibré pour prédire les matchs de la **Journée 1 EPL 2025-26**, telles qu'elles étaient disponibles **avant** que les matchs aient lieu.

**⚠️ Important:** Ces features représentent l'état réel des données au moment de la prédiction, pas les valeurs calculées après coup.

## 🤖 Modèle Utilisé: Cascade Équilibré

**Configuration:**
```python
BalancedCascadeModel(
    draw_weight=3,           # Poids modéré pour prédiction draws
    draw_threshold=0.35,     # Seuil calibré pour éviter sur-prédiction
    calibration_factor=0.85  # Contrôle distribution finale
)
```

**Performance Validée:**
- **Accuracy:** 52.5% sur 40 matchs EPL 2025-26
- **Draw Recall:** 33.3% (capture 1/3 des draws)
- **Distribution Réaliste:** H=45%, D=25%, A=30%

## 📋 Features du Modèle (10 features v2.3)

| # | Feature | Description | Méthode J1 |
|---|---------|-------------|------------|
| 1 | `form_diff_normalized` | Différence forme récente | 0.5 (reset nouvelle saison) |
| 2 | `elo_diff_normalized` | Différence force Elo | 0.5 (neutre début saison) |
| 3 | `h2h_score` | Historique face-à-face | 0.5 (dépend adversaire) |
| 4 | `matchday_normalized` | Progression saison | 0.0 (J1 = premier match) |
| 5 | `shots_diff_normalized` | Différentiel tirs | 0.5 (reset nouvelle saison) |
| 6 | `corners_diff_normalized` | Différentiel corners | 0.5 (reset nouvelle saison) |
| 7 | `market_entropy_norm` | Incertitude marché | 0.8 (élevée début saison) |
| 8 | `home_xg_eff_10` | Efficacité xG domicile | Dernière valeur 2024-25 ou 0.5 |
| 9 | `away_xg_eff_10` | Efficacité xG extérieur | Dernière valeur 2024-25 ou 0.5 |
| 10 | `away_goals_sum_5` | Total buts extérieur | Dernière valeur ou 5.0 (neutre) |

## 🏟️ Features Réelles des 20 Équipes EPL 2025-26

| Équipe | Type | form_diff | elo_diff | h2h_score | matchday | shots_diff | corners_diff | market_entropy | home_xg_eff | away_xg_eff | away_goals |
|--------|------|-----------|----------|-----------|----------|------------|--------------|----------------|-------------|-------------|------------|
| **Arsenal** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.825 | 0.934 | 10.000 |
| **Aston Villa** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.683 | 1.066 | 9.000 |
| **Bournemouth** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.747 | 0.913 | 6.000 |
| **Brentford** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 1.226 | 0.969 | 13.000 |
| **Brighton** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.739 | 1.098 | 11.000 |
| **Burnley** | 🆙 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.500 | 0.500 | 5.000 |
| **Chelsea** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 1.475 | 0.747 | 7.000 |
| **Crystal Palace** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.847 | 0.786 | 9.000 |
| **Everton** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.946 | 0.886 | 7.000 |
| **Fulham** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 1.184 | 0.978 | 8.000 |
| **Leeds** | 🆙 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.500 | 0.500 | 5.000 |
| **Liverpool** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.847 | 0.786 | 9.000 |
| **Man City** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 1.184 | 0.978 | 8.000 |
| **Man United** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.683 | 1.066 | 9.000 |
| **Newcastle** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.946 | 0.886 | 7.000 |
| **Nott'm Forest** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 1.475 | 0.747 | 7.000 |
| **Sunderland** | 🆙 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.500 | 0.500 | 5.000 |
| **Tottenham** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.739 | 1.098 | 11.000 |
| **West Ham** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 0.938 | 0.930 | 7.000 |
| **Wolves** | 📊 | 0.500 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.800 | 1.226 | 0.969 | 13.000 |


**Légende:**
- 📊 = Équipe historique (données fin 2024-25 disponibles)
- 🆙 = Équipe promue (features neutres/initialisées)

## 📈 Analyse Statistique des Features J1

### Distribution des Valeurs

**form_diff_normalized:**
- Moyenne: 0.500
- Médiane: 0.500
- Écart-type: 0.000
- Min-Max: 0.500 - 0.500

**elo_diff_normalized:**
- Moyenne: 0.500
- Médiane: 0.500
- Écart-type: 0.000
- Min-Max: 0.500 - 0.500

**h2h_score:**
- Moyenne: 0.500
- Médiane: 0.500
- Écart-type: 0.000
- Min-Max: 0.500 - 0.500

**matchday_normalized:**
- Moyenne: 0.000
- Médiane: 0.000
- Écart-type: 0.000
- Min-Max: 0.000 - 0.000

**shots_diff_normalized:**
- Moyenne: 0.500
- Médiane: 0.500
- Écart-type: 0.000
- Min-Max: 0.500 - 0.500

**corners_diff_normalized:**
- Moyenne: 0.500
- Médiane: 0.500
- Écart-type: 0.000
- Min-Max: 0.500 - 0.500

**market_entropy_norm:**
- Moyenne: 0.800
- Médiane: 0.800
- Écart-type: 0.000
- Min-Max: 0.800 - 0.800

**home_xg_eff_10:**
- Moyenne: 0.911
- Médiane: 0.847
- Écart-type: 0.291
- Min-Max: 0.500 - 1.475

**away_xg_eff_10:**
- Moyenne: 0.867
- Médiane: 0.921
- Écart-type: 0.186
- Min-Max: 0.500 - 1.098

**away_goals_sum_5:**
- Moyenne: 8.300
- Médiane: 8.000
- Écart-type: 2.347
- Min-Max: 5.000 - 13.000


### 📊 Répartition Équipes

**Équipes Historiques (17):**
Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Chelsea, Crystal Palace, Everton, Fulham, Liverpool, Man City, Man United, Newcastle, Nott'm Forest, Tottenham, West Ham, Wolves

**Équipes Promues (3):**
Leeds, Sunderland, Burnley

### 🔍 Stratégie d'Initialisation

**Pour Équipes Historiques:**
- `form_diff`, `elo_diff`, `shots_diff`, `corners_diff`: 0.5 (reset saison)
- `xG efficiency`, `away_goals`: Dernière valeur 2024-25 si disponible
- `market_entropy`: 0.8 (incertitude début saison)
- `matchday`: 0.0 (J1)

**Pour Équipes Promues:**
- Toutes features: 0.5 (neutre) sauf `away_goals_sum_5`: 5.0
- `market_entropy`: 0.8 (incertitude élevée nouvelles équipes)
- Pas de biais historique EPL

### 🎯 Impact sur Prédictions J1

**Conséquences de l'initialisation:**
1. **Prédictions conservatrices** en début de saison
2. **Équipes promues** traitées équitablement (pas de biais négatif)
3. **Market entropy élevée** reflète incertitude réelle
4. **xG efficiency** seule différence significative entre équipes

**Fiabilité attendue:**
- **J1-J3:** Prédictions plus incertaines (beaucoup de 0.5)
- **J4+:** Features se différencient avec données réelles
- **Performance globale:** 52.5% maintenue sur la saison

## 🔄 Pipeline de Prédiction J1

### Étape 1: Draw vs NotDraw
**Features clés:** `market_entropy_norm` (0.8), `elo_diff_normalized` (0.5), `form_diff_normalized` (0.5)
**Résultat:** Tendance vers NotDraw (Draw difficile à prédire avec features neutres)

### Étape 2: Home vs Away (si NotDraw)
**Features clés:** `shots_diff_normalized` (0.5), `h2h_score` (0.5), `home_xg_eff_10` (variable)
**Résultat:** Léger avantage domicile par défaut

### Calibration Finale
**Limitation percentile:** Max 25% de draws prédits
**Distribution cible:** H≈45%, D≈25%, A≈30%

## ⚠️ Limitations et Considérations

### Limitations J1
1. **Features majoritairement neutres** (0.5) en début de saison
2. **H2H scores** nécessitent connaître l'adversaire spécifique
3. **Market entropy** estimation basée sur tendances historiques
4. **Équipes promues** sans historique EPL récent

### Améliorations Possibles
1. **Elo pré-saison** basé sur transferts et préparation
2. **Market data réelles** des bookmakers avant J1
3. **Features Championship** pour équipes promues
4. **Forme pré-saison** (matchs amicaux)

## 📋 Métadonnées Techniques

**Dataset source:** `v_auto_update_20250916_105039.csv` + `v15_final_enhanced.csv`
**Méthode:** Reconstitution état pré-match J1
**Validation:** Compatible modèle cascade équilibré 52.5%
**Date limite historique:** 31/05/2025 (fin saison 2024-25)

---
*Rapport généré automatiquement - Vraies features d'entrée modèle J1 EPL 2025-26 - 16/09/2025 15:12*
