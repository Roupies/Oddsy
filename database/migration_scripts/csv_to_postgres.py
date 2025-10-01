#!/usr/bin/env python3
"""
📊 CSV to PostgreSQL Migration Script
====================================

Migration optimisée des données CSV vers PostgreSQL avec COPY.
Plus rapide que les INSERT individuels pour de gros volumes.
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging
from pathlib import Path
import io
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_connector import OddsyDatabase

class CSVToPostgresqlMigrator:
    """Migrateur CSV vers PostgreSQL avec COPY optimisé"""
    
    def __init__(self, db: OddsyDatabase):
        self.db = db
        self.conn = db.conn
        
    def migrate_teams_from_csv(self, csv_path: str):
        """Migre les équipes depuis un CSV"""
        logging.info(f"📊 Migration équipes: {csv_path}")
        
        # Lire le CSV
        df = pd.read_csv(csv_path)
        
        # Préparer les données pour COPY
        # Format attendu: team_name, short_name, league
        if 'team_name' not in df.columns:
            # Si colonnes différentes, mapper
            df = self._standardize_teams_columns(df)
        
        # Utiliser COPY pour insertion rapide
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
        output.seek(0)
        
        with self.conn.cursor() as cursor:
            try:
                # Vider la table d'abord (optionnel)
                cursor.execute("TRUNCATE TABLE teams RESTART IDENTITY CASCADE;")
                
                # COPY ultra-rapide
                cursor.copy_expert(
                    "COPY teams (team_name, short_name, league) FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', NULL '\\N')",
                    output
                )
                
                self.conn.commit()
                logging.info(f"✅ {len(df)} équipes migrées avec COPY")
                
            except Exception as e:
                self.conn.rollback()
                logging.error(f"❌ Erreur migration équipes: {e}")
                raise
    
    def migrate_matches_from_csv(self, csv_path: str):
        """Migre les matchs depuis CSV avec COPY optimisé"""
        logging.info(f"📊 Migration matchs: {csv_path}")
        
        # Lire le CSV principal
        df = pd.read_csv(csv_path)
        logging.info(f"📈 {len(df)} lignes trouvées dans {csv_path}")
        
        # Standardiser les colonnes pour le schéma PostgreSQL
        df_processed = self._standardize_matches_columns(df)
        
        # Valider les données
        df_processed = self._validate_matches_data(df_processed)
        logging.info(f"✅ {len(df_processed)} lignes validées")
        
        # Utiliser COPY pour performance maximale
        self._copy_matches_to_postgres(df_processed)
    
    def _standardize_matches_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise les colonnes CSV vers schéma PostgreSQL - ADAPTÉ E0 FORMAT"""
        
        # Mapping des colonnes E0 CSV (132 colonnes) vers PostgreSQL
        column_mapping = {
            'Date': 'match_date',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'home_goals',
            'FTAG': 'away_goals', 
            'FTR': 'full_time_result',
            'B365H': 'home_odds',
            'B365D': 'draw_odds',
            'B365A': 'away_odds',
            'HS': 'home_shots',
            'AS': 'away_shots',
            'HST': 'home_shots_target',
            'AST': 'away_shots_target',
            'HC': 'home_corners',
            'AC': 'away_corners'
        }
        
        # Ajouter colonne season pour E0 (inférer depuis date)
        if 'Date' in df.columns:
            df['Season'] = self._infer_season_from_date(df['Date'])
            column_mapping['Season'] = 'season'
        
        # Renommer colonnes si elles existent
        df_renamed = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Ajouter colonnes manquantes
        required_columns = [
            'match_date', 'season', 'home_team', 'away_team',
            'home_goals', 'away_goals', 'full_time_result',
            'home_odds', 'draw_odds', 'away_odds'
        ]
        
        for col in required_columns:
            if col not in df_renamed.columns:
                df_renamed[col] = None
        
        # Convertir les dates
        if 'match_date' in df_renamed.columns:
            df_renamed['match_date'] = pd.to_datetime(df_renamed['match_date']).dt.date
        
        # Mapper les IDs d'équipes depuis la base
        df_with_ids = self._add_team_ids(df_renamed)
        
        return df_with_ids
    
    def _add_team_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les team_id en lookup depuis la table teams"""
        
        # Récupérer le mapping team_name -> team_id
        teams_df = self.db.get_teams()
        team_id_map = dict(zip(teams_df['team_name'], teams_df['team_id']))
        
        # Mapper les IDs
        df['home_team_id'] = df['home_team'].map(team_id_map)
        df['away_team_id'] = df['away_team'].map(team_id_map)
        
        # Vérifier les équipes non trouvées
        missing_home = df[df['home_team_id'].isna()]['home_team'].unique()
        missing_away = df[df['away_team_id'].isna()]['away_team'].unique()
        
        if len(missing_home) > 0 or len(missing_away) > 0:
            logging.warning(f"⚠️ Équipes non trouvées: {set(missing_home) | set(missing_away)}")
        
        return df
    
    def _validate_matches_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide et nettoie les données de matchs"""
        
        # Supprimer lignes avec team_ids manquants
        initial_count = len(df)
        df_clean = df.dropna(subset=['home_team_id', 'away_team_id'])
        
        if len(df_clean) < initial_count:
            logging.warning(f"⚠️ {initial_count - len(df_clean)} lignes supprimées (équipes manquantes)")
        
        # Valider FTR (permettre NaN pour matchs futurs)
        df_clean = df_clean[
            df_clean['full_time_result'].isin(['H', 'D', 'A']) | 
            df_clean['full_time_result'].isna()
        ]
        
        # Convertir types
        numeric_columns = ['home_goals', 'away_goals', 'home_shots', 'away_shots', 
                          'home_shots_target', 'away_shots_target', 'home_corners', 'away_corners']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def _copy_matches_to_postgres(self, df: pd.DataFrame):
        """Utilise COPY pour insertion rapide des matchs"""
        
        # Sélectionner colonnes dans l'ordre de la table
        columns_order = [
            'match_date', 'season', 'home_team_id', 'away_team_id',
            'full_time_result', 'home_goals', 'away_goals',
            'home_odds', 'draw_odds', 'away_odds',
            'home_shots', 'away_shots', 'home_shots_target', 'away_shots_target',
            'home_corners', 'away_corners'
        ]
        
        # Filtrer colonnes existantes
        available_columns = [col for col in columns_order if col in df.columns]
        df_ordered = df[available_columns].copy()
        
        # Convertir les colonnes numériques en entiers pour éviter le format float
        integer_columns = ['home_team_id', 'away_team_id', 'home_goals', 'away_goals', 
                          'home_shots', 'away_shots', 'home_shots_target', 'away_shots_target',
                          'home_corners', 'away_corners']
        
        for col in integer_columns:
            if col in df_ordered.columns:
                df_ordered[col] = df_ordered[col].astype('Int64')
        
        # Préparer pour COPY
        output = io.StringIO()
        df_ordered.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
        output.seek(0)
        
        # Construire requête COPY
        copy_sql = f"COPY matches ({', '.join(available_columns)}) FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', NULL '\\N')"
        
        with self.conn.cursor() as cursor:
            try:
                # Option: vider table d'abord
                # cursor.execute("TRUNCATE TABLE matches RESTART IDENTITY CASCADE;")
                
                # COPY ultra-performant
                cursor.copy_expert(copy_sql, output)
                
                self.conn.commit()
                logging.info(f"✅ {len(df_ordered)} matchs migrés avec COPY")
                
            except Exception as e:
                self.conn.rollback()
                logging.error(f"❌ Erreur COPY matches: {e}")
                raise
    
    def _standardize_teams_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise les colonnes pour les équipes"""
        # À adapter selon votre format CSV
        return df
    
    def migrate_all_csvs(self, data_dir: str = "../data/processed"):
        """Migre tous les CSVs du dossier data"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logging.error(f"❌ Dossier {data_dir} inexistant")
            return
        
        # Chercher le CSV principal
        csv_files = list(data_path.glob("*.csv"))
        
        if not csv_files:
            logging.error(f"❌ Aucun CSV trouvé dans {data_dir}")
            return
        
        # Prendre le plus récent (ou spécifier le nom)
        main_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        logging.info(f"📊 CSV principal détecté: {main_csv.name}")
        
        # Migrer (adapter pour E0 CSV)
        self.migrate_matches_from_csv(str(main_csv))
    
    def _infer_season_from_date(self, date_series):
        """Infère la saison depuis la colonne Date (format DD/MM/YYYY)"""
        import pandas as pd
        
        try:
            # Convertir dates (format E0: DD/MM/YYYY)
            dates = pd.to_datetime(date_series, format='%d/%m/%Y')
            
            seasons = []
            for date in dates:
                year = date.year
                month = date.month
                
                # EPL season: Août à Mai année suivante
                if month >= 8:  # Août-Décembre
                    season = f"{year}-{year+1}"
                else:  # Janvier-Mai
                    season = f"{year-1}-{year}"
                
                seasons.append(season)
            
            return seasons
            
        except Exception as e:
            logging.warning(f"⚠️ Erreur inférence saison: {e}")
            # Fallback: utiliser 2025-2026 par défaut
            return ['2025-2026'] * len(date_series)

# =============================================================
# SCRIPT PRINCIPAL
# =============================================================

def main():
    """Script principal de migration"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Connexion DB
        logging.info("🔗 Connexion à PostgreSQL...")
        db = OddsyDatabase()
        
        # Créer migrateur
        migrator = CSVToPostgresqlMigrator(db)
        
        # Migrer données
        logging.info("🚀 Début migration CSV → PostgreSQL")
        
        # Option 1: Migrer E0 CSV unifié (avec J6)
        migrator.migrate_matches_from_csv("../../data/raw/E0 (9).csv")
        
        # Option 2: Migrer depuis dossier processed (désactivé)
        # migrator.migrate_all_csvs("../../data/processed")
        
        logging.info("✅ Migration terminée avec succès!")
        
        # Stats post-migration
        matches_count = db.execute_query("SELECT COUNT(*) as count FROM matches")
        logging.info(f"📊 Total matchs en base: {matches_count.iloc[0]['count']}")
        
        db.close()
        
    except Exception as e:
        logging.error(f"❌ Erreur migration: {e}")
        raise

if __name__ == "__main__":
    main()