#!/usr/bin/env python3
"""
🧪 Test CSV Structure Only
=========================

Test uniquement la structure CSV après intégration J6
(sans PostgreSQL)
"""

import pandas as pd
import logging
from pathlib import Path

def test_csv_j6_integration():
    """Test complet structure CSV avec J6"""
    
    logging.info("🔍 Test intégration J6 dans CSV...")
    
    csv_path = "../../data/raw/E0 (9).csv"
    backup_path = None
    
    # Trouver backup le plus récent
    backup_files = list(Path("../../data/raw/").glob("E0 (9)_backup_*.csv"))
    if backup_files:
        backup_path = max(backup_files, key=lambda x: x.stat().st_mtime)
        logging.info(f"📁 Backup trouvé: {backup_path.name}")
    
    try:
        # Charger CSV actuel
        df_current = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # Charger backup si disponible
        if backup_path:
            df_backup = pd.read_csv(backup_path, encoding='utf-8-sig')
            
            logging.info(f"📊 Comparaison:")
            logging.info(f"   Backup: {len(df_backup)} lignes")
            logging.info(f"   Actuel: {len(df_current)} lignes")
            logging.info(f"   Ajoutées: {len(df_current) - len(df_backup)} lignes")
        
        # Tests structure
        assert len(df_current) >= 60, f"Expected >=60 rows, got {len(df_current)}"
        assert len(df_current.columns) == 132, f"Expected 132 columns, got {len(df_current.columns)}"
        
        # Tests colonnes essentielles
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTR']
        missing_cols = [col for col in required_cols if col not in df_current.columns]
        assert not missing_cols, f"Missing columns: {missing_cols}"
        
        # Analyser données J6
        logging.info("\\n🏆 ANALYSE DONNÉES J6:")
        
        # Détecter J6 (nouvelles données depuis backup)
        if backup_path:
            j6_start_idx = len(df_backup)
            j6_data = df_current.iloc[j6_start_idx:]
        else:
            # Fallback: dernières 15 lignes
            j6_data = df_current.tail(15)
        
        logging.info(f"📊 Données J6 détectées: {len(j6_data)} lignes")
        
        # Tests odds J6
        j6_with_odds = j6_data.dropna(subset=['B365H', 'B365D', 'B365A'])
        logging.info(f"✅ Matchs J6 avec odds: {len(j6_with_odds)}")
        
        if len(j6_with_odds) == 0:
            logging.warning("⚠️ Aucun match J6 avec odds trouvé!")
            return False
        
        # Valider odds
        odds_issues = 0
        for idx, row in j6_with_odds.iterrows():
            issues = []
            
            if not (1.01 <= row['B365H'] <= 50):
                issues.append(f"Home odds: {row['B365H']}")
            if not (1.01 <= row['B365D'] <= 50):
                issues.append(f"Draw odds: {row['B365D']}")
            if not (1.01 <= row['B365A'] <= 50):
                issues.append(f"Away odds: {row['B365A']}")
            
            if issues:
                logging.warning(f"⚠️ {row['HomeTeam']} vs {row['AwayTeam']}: {', '.join(issues)}")
                odds_issues += 1
        
        if odds_issues == 0:
            logging.info("✅ Tous les odds J6 sont valides")
        else:
            logging.warning(f"⚠️ {odds_issues} matchs avec odds invalides")
        
        # Échantillon J6
        logging.info("\\n📋 ÉCHANTILLON MATCHS J6:")
        for _, row in j6_with_odds.head(5).iterrows():
            date_str = row['Date']
            home = row['HomeTeam']
            away = row['AwayTeam']
            h_odd = row['B365H']
            d_odd = row['B365D']
            a_odd = row['B365A']
            
            # Calculer probabilités implicites
            prob_sum = 1/h_odd + 1/d_odd + 1/a_odd
            margin = (prob_sum - 1) * 100
            
            logging.info(f"   📅 {date_str}: {home} vs {away}")
            logging.info(f"      Odds: {h_odd:.2f} / {d_odd:.2f} / {a_odd:.2f}")
            logging.info(f"      Marge: {margin:.1f}%")
        
        # Tests résultats (doivent être vides pour J6)
        j6_with_results = j6_data[j6_data['FTR'].notna()]
        if len(j6_with_results) > 0:
            logging.warning(f"⚠️ {len(j6_with_results)} matchs J6 ont déjà des résultats!")
        else:
            logging.info("✅ Matchs J6 sans résultats (normal, à jouer)")
        
        # Tests équipes
        j6_teams = set(j6_data['HomeTeam'].tolist() + j6_data['AwayTeam'].tolist())
        j6_teams = {team for team in j6_teams if pd.notna(team)}
        
        logging.info(f"\\n🏟️ Équipes J6: {len(j6_teams)}")
        logging.info(f"   {', '.join(sorted(j6_teams)[:10])}{'...' if len(j6_teams) > 10 else ''}")
        
        # Statistiques finales
        logging.info("\\n" + "="*50)
        logging.info("📊 STATISTIQUES FINALES CSV E0")
        logging.info("="*50)
        logging.info(f"📏 Total lignes: {len(df_current)}")
        logging.info(f"📏 Total colonnes: {len(df_current.columns)}")
        logging.info(f"🏆 Matchs J6: {len(j6_data)}")
        logging.info(f"🎰 Odds J6 disponibles: {len(j6_with_odds)}")
        logging.info(f"⚽ Résultats J6: {len(j6_with_results)} (attendu: 0)")
        
        # Validation finale
        success = (
            len(df_current) >= 60 and
            len(j6_with_odds) >= 5 and
            odds_issues == 0 and
            len(j6_with_results) == 0
        )
        
        if success:
            logging.info("\\n✅ INTÉGRATION J6 VALIDÉE!")
            logging.info("🚀 CSV prêt pour migration PostgreSQL")
        else:
            logging.info("\\n❌ INTÉGRATION J6 ÉCHOUÉE")
        
        return success
        
    except Exception as e:
        logging.error(f"❌ Test CSV échoué: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================
# SCRIPT PRINCIPAL
# =============================================================

def main():
    """Test principal CSV seulement"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("🧪 Test structure CSV J6 (sans PostgreSQL)")
    logging.info("=" * 60)
    
    success = test_csv_j6_integration()
    
    if success:
        logging.info("\\n🎉 Test CSV réussi!")
        logging.info("💡 Prochaine étape: Démarrer Docker et tester migration PostgreSQL")
        logging.info("   docker-compose up -d")
        logging.info("   python3 test_j6_pipeline.py")
    else:
        logging.error("\\n💥 Test CSV échoué")
    
    return success

if __name__ == "__main__":
    main()