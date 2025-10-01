#!/usr/bin/env python3
"""
🔧 DATA NORMALIZER - Normalisation équipes & données
==================================================

Normalise les noms d'équipes et données provenant de différentes sources
(CSV, API) avec détection d'anomalies et logging.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import re

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_normalizer")

# Mapping équipes (extensible)
TEAM_MAPPING = {
    "Spurs": "Tottenham",
    "Man Utd": "Man United", 
    "Man United": "Man United",
    "Nott'm Forest": "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle"
}

# Équipes connues EPL 2025-26
KNOWN_EPL_TEAMS = {
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 
    'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
    'Leeds', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
    "Nott'm Forest", 'Sunderland', 'Tottenham', 'West Ham', 'Wolves'
}

class DataNormalizer:
    """Normalise données matchs EPL de différentes sources"""
    
    def __init__(self, custom_mapping=None):
        self.team_mapping = TEAM_MAPPING.copy()
        if custom_mapping:
            self.team_mapping.update(custom_mapping)
        
        self.unknown_teams = set()
        
    def normalize_team_name(self, team_name):
        """Normalise nom équipe avec détection anomalies"""
        if pd.isna(team_name):
            logger.error("❌ Nom équipe NaN détecté")
            return None
        
        team_name = str(team_name).strip()
        
        # Appliquer mapping
        normalized = self.team_mapping.get(team_name, team_name)
        
        # Vérifier si équipe connue
        if normalized not in KNOWN_EPL_TEAMS:
            if normalized not in self.unknown_teams:
                logger.warning(f"⚠️  Équipe non reconnue: '{normalized}' (original: '{team_name}')")
                self.unknown_teams.add(normalized)
        
        return normalized
    
    def parse_date(self, date_str, format_hint=None):
        """Parse date flexible avec formats multiples"""
        if pd.isna(date_str):
            return None
            
        date_str = str(date_str).strip()
        
        # Formats supportés
        formats = [
            "%d/%m/%Y %H:%M",  # 15/08/2025 20:00
            "%d/%m/%Y",        # 15/08/2025
            "%Y-%m-%d %H:%M:%S",  # 2025-08-15 20:00:00
            "%Y-%m-%d",        # 2025-08-15
        ]
        
        if format_hint:
            formats.insert(0, format_hint)
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        logger.error(f"❌ Format date non reconnu: '{date_str}'")
        return None
    
    def parse_result(self, result_str):
        """Parse résultat match: '4 - 2' → ('H', 4, 2)"""
        if pd.isna(result_str):
            return None, None, None
            
        result_str = str(result_str).strip()
        
        # Pattern: "4 - 2" ou "4-2" ou "4:2"
        pattern = r'(\d+)\s*[-:]\s*(\d+)'
        match = re.match(pattern, result_str)
        
        if not match:
            logger.error(f"❌ Format résultat non reconnu: '{result_str}'")
            return None, None, None
        
        home_goals = int(match.group(1))
        away_goals = int(match.group(2))
        
        # Déterminer résultat
        if home_goals > away_goals:
            outcome = 'H'
        elif home_goals < away_goals:
            outcome = 'A'
        else:
            outcome = 'D'
        
        return outcome, home_goals, away_goals
    
    def normalize_csv_data(self, df, source_format="auto"):
        """Normalise DataFrame complet selon source"""
        logger.info(f"🔧 Normalisation {len(df)} matchs...")
        
        df_norm = df.copy()
        
        # Détection automatique colonnes
        if source_format == "auto":
            source_format = self._detect_format(df_norm)
            logger.info(f"📋 Format détecté: {source_format}")
        
        # Normalisation selon format
        if source_format == "epl_calendar":
            df_norm = self._normalize_epl_calendar(df_norm)
        elif source_format == "football_data":
            df_norm = self._normalize_football_data(df_norm)
        else:
            logger.warning(f"⚠️  Format inconnu: {source_format}, normalisation basique")
            df_norm = self._normalize_generic(df_norm)
        
        # Validation finale
        errors = self._validate_normalized_data(df_norm)
        if errors:
            logger.error(f"❌ {len(errors)} erreurs validation:")
            for error in errors:
                logger.error(f"  • {error}")
        
        logger.info(f"✅ Normalisation terminée: {len(df_norm)} matchs valides")
        return df_norm
    
    def _detect_format(self, df):
        """Détecte format source automatiquement"""
        columns = set(df.columns.str.lower())
        
        if 'home team' in columns and 'away team' in columns:
            return "epl_calendar"
        elif 'hometeam' in columns and 'awayteam' in columns:
            return "football_data"
        else:
            return "generic"
    
    def _normalize_epl_calendar(self, df):
        """Normalise format EPL calendar"""
        df_norm = pd.DataFrame()
        
        # Colonnes standard
        df_norm['HomeTeam'] = df['Home Team'].apply(self.normalize_team_name)
        df_norm['AwayTeam'] = df['Away Team'].apply(self.normalize_team_name)
        df_norm['Date'] = df['Date'].apply(self.parse_date)
        
        # Parse résultats
        results = df['Result'].apply(self.parse_result)
        df_norm['FullTimeResult'] = [r[0] for r in results]
        df_norm['FTHG'] = [r[1] for r in results]
        df_norm['FTAG'] = [r[2] for r in results]
        
        # Métadonnées
        df_norm['Season'] = '2025-2026'
        
        return df_norm
    
    def _normalize_football_data(self, df):
        """Normalise format Football-Data.co.uk"""
        df_norm = pd.DataFrame()
        
        df_norm['HomeTeam'] = df['HomeTeam'].apply(self.normalize_team_name)
        df_norm['AwayTeam'] = df['AwayTeam'].apply(self.normalize_team_name)
        df_norm['Date'] = df['Date'].apply(self.parse_date)
        df_norm['FullTimeResult'] = df['FTR']
        df_norm['FTHG'] = df['FTHG']
        df_norm['FTAG'] = df['FTAG']
        
        # Saison depuis date
        df_norm['Season'] = df_norm['Date'].apply(
            lambda x: '2025-2026' if x and x.year == 2025 else 'Unknown'
        )
        
        return df_norm
    
    def _normalize_generic(self, df):
        """Normalisation générique basique"""
        logger.warning("🔧 Normalisation générique appliquée")
        return df
    
    def _validate_normalized_data(self, df):
        """Valide données normalisées"""
        errors = []
        
        # Vérifications essentielles
        required_cols = ['HomeTeam', 'AwayTeam', 'Date', 'FullTimeResult']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Colonnes manquantes: {missing_cols}")
        
        # Équipes nulles
        null_teams = df[df['HomeTeam'].isna() | df['AwayTeam'].isna()]
        if len(null_teams) > 0:
            errors.append(f"{len(null_teams)} matchs avec équipes nulles")
        
        # Dates nulles  
        null_dates = df[df['Date'].isna()]
        if len(null_dates) > 0:
            errors.append(f"{len(null_dates)} matchs avec dates nulles")
        
        # Doublons
        duplicates = df.duplicated(['Date', 'HomeTeam', 'AwayTeam'])
        if duplicates.any():
            errors.append(f"{duplicates.sum()} doublons détectés")
        
        return errors
    
    def get_unknown_teams_report(self):
        """Rapport équipes non reconnues"""
        if self.unknown_teams:
            logger.warning(f"📋 {len(self.unknown_teams)} équipes non reconnues:")
            for team in sorted(self.unknown_teams):
                logger.warning(f"  • {team}")
        else:
            logger.info("✅ Toutes les équipes sont reconnues")
        
        return list(self.unknown_teams)

# Interface simple
def normalize_match_data(csv_path, output_path=None):
    """Interface simple normalisation CSV"""
    logger.info(f"🚀 Normalisation: {csv_path}")
    
    # Charger données
    df = pd.read_csv(csv_path)
    
    # Normaliser
    normalizer = DataNormalizer()
    df_normalized = normalizer.normalize_csv_data(df)
    
    # Rapport équipes
    normalizer.get_unknown_teams_report()
    
    # Sauvegarder si demandé
    if output_path:
        df_normalized.to_csv(output_path, index=False)
        logger.info(f"💾 Données normalisées sauvées: {output_path}")
    
    return df_normalized

if __name__ == "__main__":
    # Test sur nouveau CSV EPL
    df_normalized = normalize_match_data(
        'data/raw/epl-2025-GMTStandardTime_NEW.csv',
        'data/processed/epl_2025_26_normalized.csv'
    )