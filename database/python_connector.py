#!/usr/bin/env python3
"""
🗄️ Oddsy Database Connector
==========================

Connecteur Python pour la base PostgreSQL d'Oddsy.
Fournit des méthodes simples pour interaction avec la base.
"""

import psycopg2
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

class OddsyDatabase:
    """Connecteur pour la base de données Oddsy PostgreSQL"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "oddsy_football", 
                 user: str = "oddsy_user",
                 password: str = "oddsy_password"):
        
        self.connection_params = {
            'host': host,
            'port': port, 
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Établit la connexion à PostgreSQL"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            logging.info("✅ Connexion PostgreSQL établie")
        except psycopg2.Error as e:
            logging.error(f"❌ Erreur connexion PostgreSQL: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Exécute une requête SELECT et retourne un DataFrame"""
        try:
            return pd.read_sql(query, self.conn, params=params)
        except Exception as e:
            logging.error(f"❌ Erreur requête: {e}")
            raise
    
    def execute_non_query(self, query: str, params: tuple = None):
        """Exécute une requête INSERT/UPDATE/DELETE"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, params)
                self.conn.commit()
                logging.info("✅ Requête exécutée avec succès")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"❌ Erreur exécution: {e}")
            raise
    
    def bulk_copy_from_dataframe(self, df: pd.DataFrame, table: str, columns: list = None):
        """Utilise COPY pour insertion rapide depuis DataFrame"""
        import io
        
        if columns is None:
            columns = df.columns.tolist()
        
        # Préparer données pour COPY
        output = io.StringIO()
        df[columns].to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
        output.seek(0)
        
        # Construire requête COPY
        copy_sql = f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', NULL '\\N')"
        
        try:
            with self.conn.cursor() as cursor:
                cursor.copy_expert(copy_sql, output)
                self.conn.commit()
                logging.info(f"✅ COPY: {len(df)} lignes insérées dans {table}")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"❌ Erreur COPY: {e}")
            raise
    
    # =============================================================
    # MÉTHODES SPÉCIFIQUES ODDSY
    # =============================================================
    
    def get_teams(self) -> pd.DataFrame:
        """Récupère toutes les équipes"""
        return self.execute_query("SELECT * FROM teams ORDER BY team_name")
    
    def get_matches(self, season: str = None, limit: int = 100) -> pd.DataFrame:
        """Récupère les matchs (avec noms d'équipes)"""
        query = "SELECT * FROM match_results"
        params = None
        
        if season:
            query += " WHERE season = %s"
            params = (season,)
        
        query += " ORDER BY match_date DESC"
        
        if limit:
            query += f" LIMIT {limit}"
            
        return self.execute_query(query, params)
    
    def get_predictions(self, model_name: str = None) -> pd.DataFrame:
        """Récupère les prédictions"""
        query = """
        SELECT p.*, mr.home_team, mr.away_team, mr.match_date, mr.full_time_result
        FROM predictions p
        JOIN match_results mr ON p.match_id = mr.match_id
        """
        params = None
        
        if model_name:
            query += " WHERE p.model_name = %s"
            params = (model_name,)
        
        query += " ORDER BY mr.match_date DESC"
        
        return self.execute_query(query, params)
    
    def get_model_performance(self) -> pd.DataFrame:
        """Récupère les performances des modèles"""
        return self.execute_query("SELECT * FROM model_performance_summary")
    
    def save_prediction(self, 
                       match_id: int,
                       model_name: str,
                       model_version: str,
                       predicted_result: str,
                       probabilities: Dict[str, float],
                       features: Dict = None):
        """Sauvegarde une prédiction"""
        
        query = """
        INSERT INTO predictions (
            match_id, model_name, model_version, predicted_result,
            probability_home, probability_draw, probability_away,
            confidence_score, features_used
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (match_id, model_name, model_version) DO UPDATE SET
            predicted_result = EXCLUDED.predicted_result,
            probability_home = EXCLUDED.probability_home,
            probability_draw = EXCLUDED.probability_draw,
            probability_away = EXCLUDED.probability_away,
            confidence_score = EXCLUDED.confidence_score,
            features_used = EXCLUDED.features_used,
            prediction_date = CURRENT_TIMESTAMP
        """
        
        confidence = max(probabilities.values())
        features_json = json.dumps(features) if features else None
        
        params = (
            match_id, model_name, model_version, predicted_result,
            probabilities.get('H', 0), probabilities.get('D', 0), probabilities.get('A', 0),
            confidence, features_json
        )
        
        self.execute_non_query(query, params)
    
    def save_model_performance(self,
                              model_name: str,
                              model_version: str,
                              metrics: Dict[str, float],
                              dataset_used: str = None,
                              hyperparameters: Dict = None):
        """Sauvegarde les métriques de performance d'un modèle"""
        
        query = """
        INSERT INTO model_performance (
            model_name, model_version, evaluation_date, dataset_used,
            accuracy, precision_home, precision_draw, precision_away,
            recall_home, recall_draw, recall_away, f1_score, log_loss,
            total_predictions, correct_predictions, hyperparameters
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            model_name, model_version, datetime.now().date(), dataset_used,
            metrics.get('accuracy'), metrics.get('precision_home'), 
            metrics.get('precision_draw'), metrics.get('precision_away'),
            metrics.get('recall_home'), metrics.get('recall_draw'), 
            metrics.get('recall_away'), metrics.get('f1_score'), 
            metrics.get('log_loss'), metrics.get('total_predictions'),
            metrics.get('correct_predictions'), 
            json.dumps(hyperparameters) if hyperparameters else None
        )
        
        self.execute_non_query(query, params)
    
    def update_prediction_results(self):
        """Met à jour les prédictions avec les résultats réels"""
        query = """
        UPDATE predictions 
        SET is_correct = (
            CASE 
                WHEN predicted_result = (
                    SELECT full_time_result 
                    FROM matches 
                    WHERE matches.match_id = predictions.match_id
                    AND full_time_result IS NOT NULL
                ) THEN true 
                ELSE false 
            END
        )
        WHERE match_id IN (
            SELECT match_id FROM matches WHERE full_time_result IS NOT NULL
        )
        """
        
        self.execute_non_query(query)
        logging.info("✅ Résultats de prédictions mis à jour")
    
    def close(self):
        """Ferme la connexion"""
        if self.conn:
            self.conn.close()
            logging.info("🔌 Connexion fermée")

# =============================================================
# EXEMPLE D'UTILISATION
# =============================================================

if __name__ == "__main__":
    # Configuration logging
    logging.basicConfig(level=logging.INFO)
    
    # Test connexion
    try:
        db = OddsyDatabase()
        
        # Récupérer quelques données
        print("🏆 Équipes:")
        teams = db.get_teams()
        print(teams.head())
        
        print("\\n⚽ Matchs récents:")
        matches = db.get_matches(limit=5)
        print(matches[['match_date', 'home_team', 'away_team', 'full_time_result']])
        
        print("\\n📊 Performance des modèles:")
        performance = db.get_model_performance()
        print(performance)
        
        # Exemple sauvegarde prédiction
        # db.save_prediction(
        #     match_id=1,
        #     model_name="Baseline Champion",
        #     model_version="v2.3",
        #     predicted_result="H",
        #     probabilities={'H': 0.55, 'D': 0.25, 'A': 0.20}
        # )
        
        db.close()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")