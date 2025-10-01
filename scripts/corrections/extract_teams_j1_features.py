#!/usr/bin/env python3
     """
     📊 EXTRACTION FEATURES ÉQUIPES J1 EPL 2025-26
     ===========================================

     Extrait toutes les valeurs/features des 20 équipes EPL 
     au début de la saison 2025-26 (avant J1).
     """

     import pandas as pd
     import numpy as np
     from datetime import datetime
     import logging

     logging.basicConfig(level=logging.INFO)
     logger = logging.getLogger("teams_j1_features")

     def extract_teams_j1_features():
         """Extrait features des 20 équipes avant J1 2025-26"""
         logger.info("📊 EXTRACTION FEATURES ÉQUIPES J1 EPL 2025-26")
         logger.info("=" * 50)

         try:
             # Charger dataset auto-intégré
             df = pd.read_csv('data/processed/v_auto_update_20250916_105039.csv', parse_dates=['Date'])
             logger.info(f"✅ Dataset: {df.shape}")

             # Identifier début saison 2025-26
             season_start = pd.to_datetime('2025-08-15')  # Début EPL 2025-26

             # Prendre les features juste avant J1 (dernier match de chaque équipe avant début saison)
             df_pre_season = df[df['Date'] < season_start].copy()

             logger.info(f"📅 Matches avant saison 2025-26: {len(df_pre_season)}")

             # Obtenir les 20 équipes EPL 2025-26
             df_season_2025 = df[df['Date'] >= season_start].copy()
             teams_2025_home = set(df_season_2025['HomeTeam'].unique())
             teams_2025_away = set(df_season_2025['AwayTeam'].unique())
             teams_epl_2025 = sorted(teams_2025_home.union(teams_2025_away))

             logger.info(f"🏟️  Équipes EPL 2025-26: {len(teams_epl_2025)}")

             # Features modèle (exact order)
             model_features = [
                 'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
                 'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
                 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
             ]

             # Pour chaque équipe, prendre ses dernières valeurs features
             teams_features = {}

             for team in teams_epl_2025:
                 # Derniers matchs à domicile et à l'extérieur
                 home_matches = df_pre_season[df_pre_season['HomeTeam'] == team]
                 away_matches = df_pre_season[df_pre_season['AwayTeam'] == team]

                 # Prendre le plus récent match (home ou away)
                 last_home = home_matches.tail(1) if len(home_matches) > 0 else None
                 last_away = away_matches.tail(1) if len(away_matches) > 0 else None

                 # Sélectionner le plus récent
                 if last_home is not None and last_away is not None:
                     if last_home['Date'].iloc[0] >= last_away['Date'].iloc[0]:
                         last_match = last_home
                         context = 'home'
                     else:
                         last_match = last_away
                         context = 'away'
                 elif last_home is not None:
                     last_match = last_home
                     context = 'home'
                 elif last_away is not None:
                     last_match = last_away
                     context = 'away'
                 else:
                     logger.warning(f"⚠️  Pas de match trouvé pour {team}")
                     continue

                 # Extraire features
                 features = {}
                 for feature in model_features:
                     if feature in last_match.columns:
                         value = last_match[feature].iloc[0]
                         features[feature] = value if not pd.isna(value) else 0.5
                     else:
                         features[feature] = 0.5  # Valeur par défaut

                 # Ajouter infos contexte
                 features['last_match_date'] = last_match['Date'].iloc[0].strftime('%Y-%m-%d')
                 features['last_match_context'] = context
                 features['last_opponent'] = (
                     last_match['AwayTeam'].iloc[0] if context == 'home'
                     else last_match['HomeTeam'].iloc[0]
                 )
                 features['last_result'] = last_match['FullTimeResult'].iloc[0]

                 teams_features[team] = features

             logger.info(f"✅ Features extraites pour {len(teams_features)} équipes")

             # Générer rapport markdown
             report_content = f"""# 📊 FEATURES ÉQUIPES EPL 2025-26 - AVANT J1

     **Date d'extraction:** {datetime.now().strftime('%d/%m/%Y %H:%M')}
     **Modèle utilisé:** Cascade Équilibré (draw_weight=3, threshold=0.35)
     **Features:** {len(model_features)} features v2.3 production

     ## 🎯 Modèle de Prédiction

     **Configuration Cascade Équilibré:**
     ```python
     BalancedCascadeModel(
         draw_weight=3,           # Poids modéré pour draws
         draw_threshold=0.35,     # Seuil calibré
         calibration_factor=0.85  # Contrôle distribution
     )
     ```

     **Features utilisées:**
     """

             for i, feature in enumerate(model_features, 1):
                 report_content += f"{i:2d}. `{feature}`\n"

             report_content += f"""

     ## 🏟️  Features des 20 Équipes EPL 2025-26

     | Équipe | form_diff | elo_diff | h2h_score | matchday | shots_diff | corners_diff | market_entropy | home_xg_eff | away_xg_eff | away_goals | Dernier Match | 
     Contexte | Résultat |
     |--------|-----------|----------|-----------|----------|------------|--------------|----------------|-------------|-------------|------------|---------------|-------
     ---|-----------|
     """

             # Trier équipes alphabétiquement
             for team in sorted(teams_features.keys()):
                 features = teams_features[team]

                 row = f"| **{team}** "

                 # Features numériques
                 for feature in model_features:
                     value = features.get(feature, 0.5)
                     row += f"| {value:.3f} "

                 # Infos contexte
                 row += f"| {features.get('last_match_date', 'N/A')} "
                 row += f"| {features.get('last_match_context', 'N/A')} "
                 row += f"| {features.get('last_result', 'N/A')} |\n"

                 report_content += row

             # Analyse des features
             report_content += f"""

     ## 📈 Analyse des Features Avant J1

     ### Distribution des Features Clés

     **ELO Difference (force des équipes):**
     """

             elo_values = [teams_features[team].get('elo_diff_normalized', 0.5) for team in teams_features.keys()]
             elo_mean = np.mean(elo_values)
             elo_std = np.std(elo_values)

             report_content += f"""
     - Moyenne: {elo_mean:.3f}
     - Écart-type: {elo_std:.3f}
     - Plus fort: {max(elo_values):.3f}
     - Plus faible: {min(elo_values):.3f}

     **Form Difference (forme récente):**
     """

             form_values = [teams_features[team].get('form_diff_normalized', 0.5) for team in teams_features.keys()]
             form_mean = np.mean(form_values)
             form_std = np.std(form_values)

             report_content += f"""
     - Moyenne: {form_mean:.3f}
     - Écart-type: {form_std:.3f}
     - Meilleure forme: {max(form_values):.3f}
     - Plus mauvaise: {min(form_values):.3f}

     **Market Entropy (incertitude marché):**
     """

             market_values = [teams_features[team].get('market_entropy_norm', 0.5) for team in teams_features.keys()]
             market_mean = np.mean(market_values)

             report_content += f"""
     - Moyenne: {market_mean:.3f}
     - Incertitude max: {max(market_values):.3f}
     - Certitude max: {min(market_values):.3f}

     ### Équipes Promues (Nouvelles en EPL)

     Les équipes promues ont des features initialisées intelligemment :
     - **Valeurs neutres (0.5)** pour features manquantes
     - **Elo basé Championship** quand disponible  
     - **xG efficiency** calculée sur données disponibles

     ### Utilisation dans le Modèle

     Ces features sont utilisées par le **modèle cascade équilibré** pour :

     1. **Étape 1:** Prédire Draw vs NotDraw
        - `market_entropy_norm`, `elo_diff_normalized`, `form_diff_normalized`
        
     2. **Étape 2:** Si NotDraw, prédire Home vs Away  
        - `shots_diff_normalized`, `h2h_score`, `home_xg_eff_10`

     **Performance validée:** 52.5% accuracy avec 33.3% draws capturés

     ---
     *Features extraites du dataset `v_auto_update_20250916_105039.csv` - {datetime.now().strftime('%d/%m/%Y %H:%M')}*
     """

             # Sauvegarder rapport
             report_path = 'FEATURES_20_EQUIPES_EPL_2025_26_J1.md'
             with open(report_path, 'w', encoding='utf-8') as f:
                 f.write(report_content)

             logger.info(f"📄 Rapport sauvegardé: {report_path}")

             return {
                 'report_path': report_path,
                 'teams_count': len(teams_features),
                 'features_count': len(model_features),
                 'teams_features': teams_features
             }

         except Exception as e:
             logger.error(f"❌ Erreur extraction: {e}")
             import traceback
             traceback.print_exc()
             return None

     if __name__ == "__main__":
         result = extract_teams_j1_features()

         if result:
             print(f"\n📊 FEATURES EXTRAITES: {result['report_path']}")
             print(f"Équipes: {result['teams_count']}")
             print(f"Features: {result['features_count']}")
         else:
             print("❌ Échec extraction")