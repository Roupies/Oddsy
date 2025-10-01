#!/usr/bin/env python3
"""
📊 GÉNÉRATEUR RAPPORT 40 MATCHS EPL 2025-26
=========================================

Génère un rapport .md détaillé des 40 matchs prédits vs réels.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("report_40_matches")

class SimpleCascadeModel:
    """Modèle cascade pour rapport"""
    
    def __init__(self):
        self.clf_draw = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        self.is_fitted = False
    
    def fit(self, X, y):
        # Convertir target si numérique
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y.copy()
        
        # Étape 1: Draw vs NotDraw
        y_draw = (y_str == 'D').astype(int)
        self.clf_draw.fit(X, y_draw)
        
        # Étape 2: Home vs Away sur NotDraw
        mask_notdraw = y_str != 'D'
        if mask_notdraw.sum() > 5:
            X_notdraw = X[mask_notdraw]
            y_homeaway = y_str[mask_notdraw].map({'H': 1, 'A': 0})
            
            # Nettoyer NaN
            valid_homeaway = y_homeaway.notna()
            X_notdraw_clean = X_notdraw[valid_homeaway]
            y_homeaway_clean = y_homeaway[valid_homeaway]
            
            if len(y_homeaway_clean) > 5:
                self.clf_homeaway.fit(X_notdraw_clean, y_homeaway_clean)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        pred_draw = self.clf_draw.predict(X)
        pred_homeaway = self.clf_homeaway.predict(X)
        
        y_pred = []
        for is_draw, home_away in zip(pred_draw, pred_homeaway):
            if is_draw == 1:
                y_pred.append('D')
            else:
                y_pred.append('H' if home_away == 1 else 'A')
        
        return np.array(y_pred)

def generate_40_matches_report():
    """Génère rapport détaillé 40 matchs"""
    logger.info("📊 GÉNÉRATION RAPPORT 40 MATCHS EPL 2025-26")
    logger.info("=" * 50)
    
    try:
        # 1. Charger dataset production
        df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
        logger.info(f"✅ Dataset production: {df.shape}")
        
        # 2. Charger vrais résultats 40 matchs
        df_real = pd.read_csv('data/raw/E0 (7).csv', encoding='utf-8-sig')
        
        # Normaliser équipes
        team_mapping = {
            'Spurs': 'Tottenham',
            "Nott'm Forest": "Nott'm Forest"
        }
        df_real['HomeTeam'] = df_real['HomeTeam'].replace(team_mapping)
        df_real['AwayTeam'] = df_real['AwayTeam'].replace(team_mapping)
        
        real_matches = df_real[['HomeTeam', 'AwayTeam', 'FTR']]
        logger.info(f"✅ {len(real_matches)} vrais résultats chargés")
        
        # 3. Target encoding
        if 'target' not in df.columns and 'FullTimeResult' in df.columns:
            df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # 4. Anti-leakage: entrainement avant 2025-08-01
        train_cutoff = pd.to_datetime('2025-08-01')
        df_train = df[df['Date'] < train_cutoff].copy()
        
        # 5. Extension avec auto-intégration pour 40 matchs
        try:
            auto_dataset = pd.read_csv('data/processed/v_auto_update_20250916_105039.csv', parse_dates=['Date'])
            auto_season_2025 = auto_dataset[auto_dataset['Date'] >= '2025-08-01'].copy()
            auto_test_candidates = auto_season_2025.head(40).copy()
            df_test = pd.merge(
                auto_test_candidates, real_matches,
                on=['HomeTeam', 'AwayTeam'],
                how='inner'
            )
            logger.info(f"🔄 Extension auto: {len(df_test)} correspondances")
        except:
            df_season_2025 = df[df['Date'] >= '2025-08-01'].copy()
            df_test_candidates = df_season_2025.head(40).copy()
            df_test = pd.merge(
                df_test_candidates, real_matches,
                on=['HomeTeam', 'AwayTeam'],
                how='inner'
            )
        
        # 6. Features modèle
        model_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        # 7. Préparer données
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        
        X_train = df_train_clean[model_features].fillna(0.5)
        y_train = df_train_clean['target']
        
        X_test = df_test[model_features].fillna(0.5)
        y_real = df_test['FTR']
        
        # 8. Entrainement et prédiction
        model = SimpleCascadeModel()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_real, y_pred)
        
        logger.info(f"🎯 Performance finale: {accuracy:.1%}")
        
        # 9. Générer rapport markdown
        report_content = f"""# 🏆 RAPPORT BASELINE 40 MATCHS EPL 2025-26

**Date du rapport:** {datetime.now().strftime('%d/%m/%Y %H:%M')}

## 📊 Performance Globale

- **Accuracy:** {accuracy:.1%} ({int(accuracy * len(y_real))}/{len(y_real)} matchs corrects)
- **Modèle:** Cascade Draw vs NotDraw → Home vs Away
- **Features:** 10 features v2.3 validées
- **Entrainement:** {len(X_train):,} matchs historiques (< 2025-08-01)
- **Test:** {len(df_test)} matchs EPL 2025-26 J1-J4

## 🎯 Distribution des Résultats

### Réalité EPL 2025-26
"""
        
        # Distribution réelle
        real_dist = y_real.value_counts(normalize=True)
        report_content += f"""
- **Home:** {real_dist.get('H', 0):.1%} ({y_real.value_counts().get('H', 0)} matchs)
- **Draw:** {real_dist.get('D', 0):.1%} ({y_real.value_counts().get('D', 0)} matchs)
- **Away:** {real_dist.get('A', 0):.1%} ({y_real.value_counts().get('A', 0)} matchs)
"""
        
        # Distribution prédictions
        pred_dist = pd.Series(y_pred).value_counts(normalize=True)
        pred_counts = pd.Series(y_pred).value_counts()
        report_content += f"""
### Prédictions Cascade
- **Home:** {pred_dist.get('H', 0):.1%} ({pred_counts.get('H', 0)} prédictions)
- **Draw:** {pred_dist.get('D', 0):.1%} ({pred_counts.get('D', 0)} prédictions)
- **Away:** {pred_dist.get('A', 0):.1%} ({pred_counts.get('A', 0)} prédictions)

## 📈 Matrice de Confusion

```
           Prédictions
Réalité    H    D    A    Total   Precision
"""
        
        # Matrice confusion
        cm = confusion_matrix(y_real, y_pred, labels=['H', 'D', 'A'])
        labels = ['H', 'D', 'A']
        
        for i, label in enumerate(labels):
            total = cm[i].sum()
            correct = cm[i][i]
            precision = correct/total if total > 0 else 0
            report_content += f"   {label}    {cm[i][0]:2d}   {cm[i][1]:2d}   {cm[i][2]:2d}    {total:2d}     {precision:.1%}\n"
        
        # Benchmarks
        report_content += f"""
```

## 📈 Comparaison Benchmarks

- **vs Random (33.3%):** +{accuracy-0.333:.1%}
- **vs Majority Class (50.0%):** +{accuracy-0.50:.1%}
- **Niveau:** {'🔥 EXCELLENT' if accuracy >= 0.55 else '✅ BON' if accuracy >= 0.50 else '⚠️  ACCEPTABLE' if accuracy >= 0.45 else '❌ INSUFFISANT'}

## 🎯 Détail des 40 Matchs

| # | Équipe Domicile | Équipe Extérieur | Réel | Prédit | Correct |
|---|----------------|-----------------|------|--------|---------|
"""
        
        # Détail matchs
        for i in range(len(df_test)):
            home = df_test.iloc[i]['HomeTeam']
            away = df_test.iloc[i]['AwayTeam']
            real = y_real.iloc[i]
            pred = y_pred[i]
            correct = "✅" if real == pred else "❌"
            
            report_content += f"| {i+1:2d} | {home} | {away} | {real} | {pred} | {correct} |\n"
        
        # Analyse par outcome
        report_content += f"""

## 🔍 Analyse par Type de Match

### Matchs à Domicile (H)
"""
        home_matches = y_real == 'H'
        home_correct = (y_real[home_matches] == pd.Series(y_pred)[home_matches]).sum()
        home_total = home_matches.sum()
        home_acc = home_correct / home_total if home_total > 0 else 0
        
        report_content += f"- **Performance:** {home_acc:.1%} ({home_correct}/{home_total})\n"
        
        # Matchs nuls
        draw_matches = y_real == 'D'
        draw_correct = (y_real[draw_matches] == pd.Series(y_pred)[draw_matches]).sum()
        draw_total = draw_matches.sum()
        draw_acc = draw_correct / draw_total if draw_total > 0 else 0
        
        report_content += f"""
### Matchs Nuls (D)
- **Performance:** {draw_acc:.1%} ({draw_correct}/{draw_total})
"""
        
        # Matchs extérieur
        away_matches = y_real == 'A'
        away_correct = (y_real[away_matches] == pd.Series(y_pred)[away_matches]).sum()
        away_total = away_matches.sum()
        away_acc = away_correct / away_total if away_total > 0 else 0
        
        report_content += f"""
### Matchs à l'Extérieur (A)
- **Performance:** {away_acc:.1%} ({away_correct}/{away_total})

## ⚠️ Observations

### Points Forts
- {"Excellente" if accuracy >= 0.55 else "Bonne" if accuracy >= 0.50 else "Acceptable"} performance globale ({accuracy:.1%})
- Anti-leakage temporel strict respecté
- {len(X_train):,} matchs d'entrainement robustes

### Points d'Attention
"""
        
        if pred_counts.get('D', 0) == 0:
            report_content += "- ⚠️ **Aucun match nul prédit** - Limitation du modèle cascade\n"
        
        if accuracy < 0.50:
            report_content += "- ⚠️ **Performance sous 50%** - Nécessite amélioration\n"
        
        report_content += f"""
## 🎯 Conclusion

La baseline Oddsy sur 40 matchs EPL 2025-26 J1-J4 atteint **{accuracy:.1%} de précision** avec le modèle cascade.

**Verdict:** {'EXCELLENT - Production ready' if accuracy >= 0.55 else 'BON - Acceptable pour production' if accuracy >= 0.50 else 'ACCEPTABLE - Nécessite optimisation' if accuracy >= 0.45 else 'INSUFFISANT - Révision requise'}

---
*Rapport généré automatiquement par Oddsy v2.3 - {datetime.now().strftime('%d/%m/%Y %H:%M')}*
"""
        
        # Sauvegarder rapport
        report_path = 'RAPPORT_BASELINE_40_MATCHS_EPL_2025_26.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 Rapport sauvegardé: {report_path}")
        
        return {
            'accuracy': accuracy,
            'report_path': report_path,
            'n_test': len(df_test),
            'correct_predictions': int(accuracy * len(y_real))
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur génération rapport: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = generate_40_matches_report()
    
    if result:
        print(f"\n📄 RAPPORT GÉNÉRÉ: {result['report_path']}")
        print(f"Performance: {result['accuracy']:.1%} ({result['correct_predictions']}/{result['n_test']})")
    else:
        print("❌ Échec génération rapport")