#!/usr/bin/env python3
"""
🧪 Test Pipeline J6 Complete
============================

Test complet de la pipeline d'intégration J6 :
CSV unifié → PostgreSQL COPY → Validation données
"""

import pandas as pd
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_connector import OddsyDatabase

def test_csv_structure():
    """Test structure CSV après intégration J6"""
    logging.info("🔍 Test structure CSV...")
    
    csv_path = "../../data/raw/E0 (9).csv"
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # Tests structure
        assert len(df) >= 60, f"Expected >=60 rows, got {len(df)}"
        assert len(df.columns) == 132, f"Expected 132 columns, got {len(df.columns)}"
        
        # Tests colonnes essentielles
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']
        missing_cols = [col for col in required_cols if col not in df.columns]
        assert not missing_cols, f"Missing columns: {missing_cols}"
        
        # Tests données J6 (dernières lignes)
        j6_rows = df.tail(10)  # Dernières 10 lignes = J6
        
        # Vérifier odds J6
        j6_with_odds = j6_rows.dropna(subset=['B365H', 'B365D', 'B365A'])
        assert len(j6_with_odds) >= 5, f"Expected >=5 J6 matches with odds, got {len(j6_with_odds)}"
        
        # Vérifier format odds
        for _, row in j6_with_odds.iterrows():
            assert 1.01 <= row['B365H'] <= 50, f"Invalid home odds: {row['B365H']}"
            assert 1.01 <= row['B365D'] <= 50, f"Invalid draw odds: {row['B365D']}"
            assert 1.01 <= row['B365A'] <= 50, f"Invalid away odds: {row['B365A']}"
        
        logging.info("✅ Structure CSV valide")
        logging.info(f"   Total lignes: {len(df)}")
        logging.info(f"   Matchs J6 avec odds: {len(j6_with_odds)}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Test CSV échoué: {e}")
        return False

def test_postgresql_connection():
    """Test connexion PostgreSQL"""
    logging.info("🐘 Test connexion PostgreSQL...")
    
    try:
        db = OddsyDatabase()
        
        # Test requête simple
        result = db.execute_query("SELECT 1 as test")
        assert len(result) == 1, "Connection test failed"
        
        logging.info("✅ Connexion PostgreSQL OK")
        db.close()
        return True
        
    except Exception as e:
        logging.error(f"❌ Connexion PostgreSQL échouée: {e}")
        logging.info("💡 Assurez-vous que Docker PostgreSQL est démarré:")
        logging.info("   docker-compose up -d")
        return False

def test_csv_migration():
    """Test migration CSV → PostgreSQL"""
    logging.info("📊 Test migration CSV...")
    
    try:
        from csv_to_postgres import CSVToPostgresqlMigrator
        
        # Connexion DB
        db = OddsyDatabase()
        migrator = CSVToPostgresqlMigrator(db)
        
        # Vider tables pour test propre
        db.execute_non_query("TRUNCATE TABLE matches RESTART IDENTITY CASCADE")
        db.execute_non_query("TRUNCATE TABLE teams RESTART IDENTITY CASCADE")
        
        # Insérer équipes de base (nécessaire pour foreign keys)
        teams_data = [
            ('Liverpool', 'LIV'), ('Arsenal', 'ARS'), ('Man City', 'MCI'),
            ('Chelsea', 'CHE'), ('Tottenham', 'TOT'), ('Man United', 'MUN'),
            ('Newcastle', 'NEW'), ('Aston Villa', 'AVL'), ('Brighton', 'BHA'),
            ('West Ham', 'WHU'), ('Fulham', 'FUL'), ('Brentford', 'BRE'),
            ('Crystal Palace', 'CRY'), ('Bournemouth', 'BOU'), ('Wolves', 'WOL'),
            ('Everton', 'EVE'), ('Leicester', 'LEI'), ('Leeds', 'LEE'),
            ('Sunderland', 'SUN'), ('Burnley', 'BUR'), ("Nott'm Forest", 'NFO'),
            ('Southampton', 'SOU')
        ]
        
        for team_name, short_name in teams_data:
            db.execute_non_query(
                "INSERT INTO teams (team_name, short_name) VALUES (%s, %s) ON CONFLICT (team_name) DO NOTHING",
                (team_name, short_name)
            )
        
        logging.info("✅ Équipes insérées")
        
        # Migrer matches depuis E0 CSV
        migrator.migrate_matches_from_csv("../../data/raw/E0 (9).csv")
        
        # Vérifier résultats
        matches_count = db.execute_query("SELECT COUNT(*) as count FROM matches")
        total_matches = matches_count.iloc[0]['count']
        
        assert total_matches >= 58, f"Expected >=58 matches, got {total_matches}"
        
        # Vérifier données J6 spécifiquement
        j6_matches = db.execute_query("""
            SELECT COUNT(*) as count 
            FROM matches 
            WHERE match_date >= '2025-09-27' 
            AND home_odds IS NOT NULL
        """)
        j6_count = j6_matches.iloc[0]['count']
        
        assert j6_count >= 5, f"Expected >=5 J6 matches with odds, got {j6_count}"
        
        logging.info("✅ Migration COPY réussie")
        logging.info(f"   Total matchs migrés: {total_matches}")
        logging.info(f"   Matchs J6 avec odds: {j6_count}")
        
        db.close()
        return True
        
    except Exception as e:
        logging.error(f"❌ Migration échouée: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_integrity():
    """Test intégrité des données migrées"""
    logging.info("🔍 Test intégrité données...")
    
    try:
        db = OddsyDatabase()
        
        # Test 1: Pas de doublons
        duplicates = db.execute_query("""
            SELECT match_date, home_team_id, away_team_id, COUNT(*)
            FROM matches 
            GROUP BY match_date, home_team_id, away_team_id 
            HAVING COUNT(*) > 1
        """)
        
        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate matches"
        
        # Test 2: Odds cohérents
        invalid_odds = db.execute_query("""
            SELECT COUNT(*) as count
            FROM matches 
            WHERE home_odds IS NOT NULL 
            AND (home_odds < 1.01 OR home_odds > 50 
                 OR draw_odds < 1.01 OR draw_odds > 50
                 OR away_odds < 1.01 OR away_odds > 50)
        """)
        
        assert invalid_odds.iloc[0]['count'] == 0, f"Found {invalid_odds.iloc[0]['count']} matches with invalid odds"
        
        # Test 3: Données J6
        j6_data = db.execute_query("""
            SELECT home_team, away_team, home_odds, draw_odds, away_odds
            FROM match_results 
            WHERE match_date >= '2025-09-27'
            ORDER BY match_date DESC
            LIMIT 5
        """)
        
        assert len(j6_data) >= 5, "J6 data not found in database"
        
        logging.info("✅ Intégrité données validée")
        logging.info(f"   Aucun doublon détecté")
        logging.info(f"   Odds valides pour tous les matchs")
        logging.info(f"   {len(j6_data)} matchs J6 vérifiés")
        
        # Afficher échantillon J6
        logging.info("📊 Échantillon J6:")
        for _, row in j6_data.head(3).iterrows():
            logging.info(f"   {row['home_team']} vs {row['away_team']}: "
                        f"{row['home_odds']:.2f}/{row['draw_odds']:.2f}/{row['away_odds']:.2f}")
        
        db.close()
        return True
        
    except Exception as e:
        logging.error(f"❌ Test intégrité échoué: {e}")
        return False

def run_complete_pipeline_test():
    """Execute pipeline complète avec tous les tests"""
    logging.info("🚀 DÉMARRAGE TEST PIPELINE J6 COMPLÈTE")
    logging.info("=" * 60)
    
    tests = [
        ("Structure CSV", test_csv_structure),
        ("Connexion PostgreSQL", test_postgresql_connection),
        ("Migration COPY", test_csv_migration),
        ("Intégrité données", test_data_integrity)
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        logging.info(f"\\n🧪 {test_name}...")
        try:
            success = test_func()
            results[test_name] = "✅ PASS" if success else "❌ FAIL"
            if not success:
                all_passed = False
        except Exception as e:
            results[test_name] = f"💥 ERROR: {str(e)}"
            all_passed = False
    
    # Rapport final
    logging.info("\\n" + "=" * 60)
    logging.info("📊 RAPPORT FINAL TESTS J6")
    logging.info("=" * 60)
    
    for test_name, result in results.items():
        logging.info(f"   {test_name}: {result}")
    
    if all_passed:
        logging.info("\\n🏆 PIPELINE J6 VALIDÉE - PRODUCTION READY!")
        logging.info("✅ CSV unifié → PostgreSQL COPY → Validation complète")
    else:
        logging.info("\\n❌ PIPELINE J6 ÉCHOUÉE - Corrections nécessaires")
    
    logging.info("=" * 60)
    return all_passed

# =============================================================
# SCRIPT PRINCIPAL
# =============================================================

def main():
    """Test principal pipeline J6"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    success = run_complete_pipeline_test()
    
    if success:
        logging.info("🎉 Tous les tests passés !")
        exit(0)
    else:
        logging.error("💥 Certains tests ont échoué")
        exit(1)

if __name__ == "__main__":
    main()