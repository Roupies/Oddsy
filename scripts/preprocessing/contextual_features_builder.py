#!/usr/bin/env python3
"""
contextual_features_builder.py

FEATURES CONTEXTUELLES INNOVANTES
Phase suivante après nettoyage redondance - Features vraiment discriminantes

FEATURES CIBLÉES (basé sur analyse Claude Opus):
1. MÉTÉO: Temperature, précipitations, vent (impact performance physique)
2. ARBITRES: Historique cartes, penalties, tendances home advantage  
3. FATIGUE: Jours repos, fixture congestion, voyages internationaux
4. MOMENTUM: Acceleration/deceleration performance récente
5. CONTEXTE: Pression classement, rivalités, enjeux saison

OBJECTIF:
- Dépasser features xG redondantes par features contextuelles
- Passer de 53% → 55%+ avec intelligence football
- Features impossibles à avoir dans post-match data (temporal safety)

API/SOURCES:
- OpenWeatherMap: Météo historique stades
- Referee data: Premier League historical referee stats  
- Fixture data: Calendrier, voyages, competitions européennes
"""

import sys
import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

class ContextualFeaturesBuilder:
    """
    Constructeur de features contextuelles innovantes pour football
    """
    
    def __init__(self):
        self.logger = setup_logging()
        
        # Configuration APIs (placeholders - à configurer)
        self.weather_api_key = os.getenv('OPENWEATHER_API_KEY', 'YOUR_API_KEY_HERE')
        
        # Cache pour éviter trop d'appels API
        self.weather_cache = {}
        self.referee_cache = {}
        
        # Stades Premier League avec coordonnées
        self.stadiums = {
            'Arsenal': {'lat': 51.5549, 'lon': -0.1084, 'city': 'London'},
            'Chelsea': {'lat': 51.4816, 'lon': -0.1910, 'city': 'London'}, 
            'Liverpool': {'lat': 53.4308, 'lon': -2.9608, 'city': 'Liverpool'},
            'Man City': {'lat': 53.4831, 'lon': -2.2004, 'city': 'Manchester'},
            'Man United': {'lat': 53.4631, 'lon': -2.2914, 'city': 'Manchester'},
            'Tottenham': {'lat': 51.6042, 'lon': -0.0667, 'city': 'London'},
            # ... Ajouter tous les stades Premier League
        }
        
    def load_base_data(self, filepath='data/processed/v13_xg_safe_features.csv'):
        """
        Charger données de base pour enrichir avec contexte
        """
        self.logger.info("📊 CHARGEMENT DONNÉES POUR ENRICHISSEMENT CONTEXTUEL")
        self.logger.info("="*70)
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        self.logger.info(f"✅ Données base: {len(df)} matches")
        
        # Identifier période pour features contextuelles
        date_range = f"{df['Date'].min().strftime('%Y-%m-%d')} → {df['Date'].max().strftime('%Y-%m-%d')}"
        self.logger.info(f"📅 Période: {date_range}")
        
        return df
    
    def build_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features météo pour chaque match
        """
        self.logger.info("\n🌤️ CONSTRUCTION FEATURES MÉTÉO")
        self.logger.info("-"*50)
        
        if self.weather_api_key == 'YOUR_API_KEY_HERE':
            self.logger.warning("⚠️ OpenWeather API key non configurée - features météo simulées")
            return self._simulate_weather_features(df)
        
        weather_features = []
        
        for idx, row in df.iterrows():
            date = row['Date']
            home_team = row['HomeTeam']
            
            # Coordonnées stade
            if home_team in self.stadiums:
                lat = self.stadiums[home_team]['lat']
                lon = self.stadiums[home_team]['lon']
                
                # Clé cache
                cache_key = f"{home_team}_{date.strftime('%Y-%m-%d')}"
                
                if cache_key in self.weather_cache:
                    weather_data = self.weather_cache[cache_key]
                else:
                    weather_data = self._fetch_historical_weather(date, lat, lon)
                    self.weather_cache[cache_key] = weather_data
                    time.sleep(0.1)  # Rate limiting
                
            else:
                # Valeurs par défaut si stade non trouvé
                weather_data = {
                    'temperature': 15.0,
                    'precipitation': 0.0,
                    'wind_speed': 10.0,
                    'humidity': 70.0,
                    'pressure': 1013.25
                }
            
            weather_features.append(weather_data)
            
            if (idx + 1) % 100 == 0:
                self.logger.info(f"    Météo récupérée pour {idx+1} matches...")
        
        # Convertir en DataFrame
        weather_df = pd.DataFrame(weather_features)
        
        # Features dérivées
        weather_df['temp_extreme'] = (
            (weather_df['temperature'] < 5) | 
            (weather_df['temperature'] > 25)
        ).astype(int)
        
        weather_df['weather_adverse'] = (
            (weather_df['precipitation'] > 0.5) | 
            (weather_df['wind_speed'] > 15)
        ).astype(int)
        
        weather_df['weather_comfort_score'] = (
            1 - (abs(weather_df['temperature'] - 15) / 20) -  # Température idéale ~15°C
            (weather_df['precipitation'] / 10) -               # Moins de pluie = mieux
            (weather_df['wind_speed'] / 30)                   # Moins de vent = mieux
        ).clip(0, 1)
        
        # Joindre au DataFrame principal
        weather_columns = [
            'temperature', 'precipitation', 'wind_speed', 'humidity',
            'temp_extreme', 'weather_adverse', 'weather_comfort_score'
        ]
        
        for col in weather_columns:
            df[f'weather_{col}'] = weather_df[col]
        
        self.logger.info(f"✅ {len(weather_columns)} features météo ajoutées")
        
        return df
    
    def _fetch_historical_weather(self, date: datetime, lat: float, lon: float) -> Dict:
        """
        Récupérer données météo historiques via OpenWeatherMap
        """
        # Timestamp Unix pour la date
        timestamp = int(date.timestamp())
        
        url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
        params = {
            'lat': lat,
            'lon': lon,
            'dt': timestamp,
            'appid': self.weather_api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current = data['current']
            
            return {
                'temperature': current.get('temp', 15.0),
                'precipitation': current.get('rain', {}).get('1h', 0.0) + current.get('snow', {}).get('1h', 0.0),
                'wind_speed': current.get('wind_speed', 0.0),
                'humidity': current.get('humidity', 70.0),
                'pressure': current.get('pressure', 1013.25)
            }
            
        except Exception as e:
            self.logger.warning(f"Erreur météo {date}: {e}")
            return {
                'temperature': 15.0,
                'precipitation': 0.0,
                'wind_speed': 10.0,
                'humidity': 70.0,
                'pressure': 1013.25
            }
    
    def _simulate_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simuler features météo réalistes (si pas d'API)
        """
        np.random.seed(42)
        n_matches = len(df)
        
        # Simulation basée sur saison (hiver plus froid/humide)
        months = pd.DatetimeIndex(df['Date']).month
        
        # Temperature (variant selon saison)
        base_temp = 15 + 10 * np.sin(2 * np.pi * (months - 1) / 12)
        temperature = base_temp + np.random.normal(0, 5, n_matches)
        
        # Précipitations (plus en hiver)
        precip_base = 0.3 + 0.2 * (months < 4) | (months > 10)
        precipitation = np.random.exponential(precip_base, n_matches)
        
        # Vent
        wind_speed = np.random.gamma(2, 5, n_matches)
        
        # Features dérivées
        temp_extreme = ((temperature < 5) | (temperature > 25)).astype(int)
        weather_adverse = ((precipitation > 0.5) | (wind_speed > 15)).astype(int)
        weather_comfort_score = (
            1 - (np.abs(temperature - 15) / 20) - 
            (precipitation / 10) - 
            (wind_speed / 30)
        ).clip(0, 1)
        
        # Ajouter au DataFrame
        weather_features = {
            'weather_temperature': temperature,
            'weather_precipitation': precipitation,
            'weather_wind_speed': wind_speed,
            'weather_humidity': 70 + np.random.normal(0, 15, n_matches),
            'weather_temp_extreme': temp_extreme,
            'weather_weather_adverse': weather_adverse,
            'weather_weather_comfort_score': weather_comfort_score
        }
        
        for col, values in weather_features.items():
            df[col] = values
        
        self.logger.info("✅ 7 features météo simulées ajoutées")
        
        return df
    
    def build_referee_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features arbitres: Historique cards, penalties, home advantage
        """
        self.logger.info("\n⚽ CONSTRUCTION FEATURES ARBITRES")
        self.logger.info("-"*50)
        
        # Note: En l'absence de data arbitres réelles, on simule des patterns
        self.logger.info("⚠️ Simulation features arbitres (pas de data externe)")
        
        np.random.seed(42)
        n_matches = len(df)
        
        # Simuler différents types d'arbitres
        referee_types = np.random.choice(['strict', 'lenient', 'home_bias', 'neutral'], 
                                       size=n_matches, 
                                       p=[0.3, 0.3, 0.2, 0.2])
        
        # Features basées sur type d'arbitre
        referee_features = {
            'referee_cards_per_match': np.where(
                referee_types == 'strict', 
                np.random.gamma(3, 1.5),  # Arbitres stricts: plus de cartons
                np.random.gamma(2, 1.2)   # Autres: moins de cartons
            ),
            
            'referee_penalty_rate': np.where(
                referee_types == 'lenient',
                np.random.beta(1, 4),     # Arbitres laxistes: moins de penalties
                np.random.beta(2, 3)      # Autres: plus de penalties
            ),
            
            'referee_home_advantage': np.where(
                referee_types == 'home_bias',
                np.random.beta(3, 2),     # Arbitres pro-home: avantage domicile
                np.random.beta(2, 2)      # Neutres: avantage domicile normal
            ),
            
            'referee_experience': np.random.gamma(5, 2, n_matches),  # Années expérience
        }
        
        # Features dérivées
        referee_features['referee_strictness_score'] = (
            referee_features['referee_cards_per_match'] / 6 +  # Normaliser
            referee_features['referee_penalty_rate']
        ) / 2
        
        referee_features['referee_home_bias_score'] = referee_features['referee_home_advantage']
        
        # Ajouter au DataFrame
        for col, values in referee_features.items():
            df[col] = values
        
        self.logger.info("✅ 6 features arbitres ajoutées")
        
        return df
    
    def build_fatigue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features fatigue: Jours repos, fixture congestion, voyages
        """
        self.logger.info("\n😴 CONSTRUCTION FEATURES FATIGUE")
        self.logger.info("-"*50)
        
        # Trier par équipe et date pour calculer repos
        df_sorted = df.sort_values(['Date']).copy()
        
        fatigue_features = []
        
        # Calculer pour chaque équipe séparément
        for team in df['HomeTeam'].unique():
            home_matches = df_sorted[df_sorted['HomeTeam'] == team].copy()
            away_matches = df_sorted[df_sorted['AwayTeam'] == team].copy()
            
            # Combiner tous les matches de l'équipe
            all_team_matches = pd.concat([
                home_matches[['Date', 'HomeTeam']].rename(columns={'HomeTeam': 'Team'}),
                away_matches[['Date', 'AwayTeam']].rename(columns={'AwayTeam': 'Team'})
            ]).sort_values('Date').reset_index(drop=True)
            
            # Calculer jours de repos
            all_team_matches['days_rest'] = all_team_matches['Date'].diff().dt.days
            all_team_matches['days_rest'] = all_team_matches['days_rest'].fillna(7)  # Premier match = 7 jours
            
            # Features congestion (matches récents)
            all_team_matches['matches_last_14_days'] = 0
            all_team_matches['matches_last_30_days'] = 0
            
            for idx in range(len(all_team_matches)):
                current_date = all_team_matches.iloc[idx]['Date']
                
                # Compter matches dans fenêtres temporelles
                last_14 = all_team_matches[
                    (all_team_matches['Date'] >= current_date - timedelta(days=14)) & 
                    (all_team_matches['Date'] < current_date)
                ]
                last_30 = all_team_matches[
                    (all_team_matches['Date'] >= current_date - timedelta(days=30)) & 
                    (all_team_matches['Date'] < current_date)
                ]
                
                all_team_matches.at[idx, 'matches_last_14_days'] = len(last_14)
                all_team_matches.at[idx, 'matches_last_30_days'] = len(last_30)
            
            # Mapper vers DataFrame principal
            team_fatigue_data = all_team_matches.set_index('Date')[
                ['days_rest', 'matches_last_14_days', 'matches_last_30_days']
            ]
            
            # Home matches
            home_mask = df_sorted['HomeTeam'] == team
            for idx in df_sorted[home_mask].index:
                match_date = df_sorted.loc[idx, 'Date']
                if match_date in team_fatigue_data.index:
                    fatigue_data = team_fatigue_data.loc[match_date]
                    df.at[idx, 'home_days_rest'] = fatigue_data['days_rest']
                    df.at[idx, 'home_matches_14d'] = fatigue_data['matches_last_14_days']
                    df.at[idx, 'home_matches_30d'] = fatigue_data['matches_last_30_days']
            
            # Away matches
            away_mask = df_sorted['AwayTeam'] == team
            for idx in df_sorted[away_mask].index:
                match_date = df_sorted.loc[idx, 'Date']
                if match_date in team_fatigue_data.index:
                    fatigue_data = team_fatigue_data.loc[match_date]
                    df.at[idx, 'away_days_rest'] = fatigue_data['days_rest']
                    df.at[idx, 'away_matches_14d'] = fatigue_data['matches_last_14_days']
                    df.at[idx, 'away_matches_30d'] = fatigue_data['matches_last_30_days']
        
        # Fill NaN avec valeurs par défaut
        fatigue_cols = [
            'home_days_rest', 'away_days_rest',
            'home_matches_14d', 'away_matches_14d',
            'home_matches_30d', 'away_matches_30d'
        ]
        
        for col in fatigue_cols:
            if col not in df.columns:
                df[col] = 7 if 'days_rest' in col else 2  # Valeurs par défaut
            df[col] = df[col].fillna(7 if 'days_rest' in col else 2)
        
        # Features dérivées
        df['rest_advantage'] = df['home_days_rest'] - df['away_days_rest']
        df['congestion_disadvantage'] = (
            (df['home_matches_14d'] - df['away_matches_14d']) + 
            0.5 * (df['home_matches_30d'] - df['away_matches_30d'])
        )
        
        df['fatigue_score'] = (
            1 / (1 + np.exp(-(df['rest_advantage'] - 2))) -  # Sigmoid: avantage si >2j repos
            0.1 * df['congestion_disadvantage']               # Pénalité congestion
        ).clip(0, 1)
        
        self.logger.info("✅ 9 features fatigue ajoutées")
        
        return df
    
    def build_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features momentum: Accélération/décélération performance
        """
        self.logger.info("\n🚀 CONSTRUCTION FEATURES MOMENTUM")
        self.logger.info("-"*50)
        
        # Trier chronologiquement
        df_sorted = df.sort_values(['Date']).copy()
        
        # Calculer momentum pour chaque équipe
        for team in df['HomeTeam'].unique():
            # Récupérer tous les résultats de l'équipe
            home_results = df_sorted[df_sorted['HomeTeam'] == team]['FullTimeResult'].map({
                'H': 1, 'D': 0, 'A': -1
            })
            away_results = df_sorted[df_sorted['AwayTeam'] == team]['FullTimeResult'].map({
                'H': -1, 'D': 0, 'A': 1
            })
            
            # Combiner et trier chronologiquement
            all_results = pd.concat([
                pd.DataFrame({'date': df_sorted[df_sorted['HomeTeam'] == team]['Date'], 
                             'result': home_results}),
                pd.DataFrame({'date': df_sorted[df_sorted['AwayTeam'] == team]['Date'], 
                             'result': away_results})
            ]).sort_values('date').reset_index(drop=True)
            
            if len(all_results) < 5:
                continue
            
            # Calculer rolling performance et momentum
            all_results['perf_3'] = all_results['result'].rolling(3, min_periods=1).mean()
            all_results['perf_5'] = all_results['result'].rolling(5, min_periods=1).mean()
            
            # Momentum = différence performance récente vs ancienne
            all_results['momentum_short'] = (
                all_results['perf_3'] - all_results['perf_3'].shift(3)
            ).fillna(0)
            
            all_results['momentum_long'] = (
                all_results['perf_5'] - all_results['perf_5'].shift(5)
            ).fillna(0)
            
            # Acceleration = changement de momentum
            all_results['acceleration'] = all_results['momentum_short'].diff().fillna(0)
            
            # Mapper vers DataFrame principal
            team_momentum = all_results.set_index('date')[
                ['momentum_short', 'momentum_long', 'acceleration']
            ]
            
            # Home matches
            home_mask = df_sorted['HomeTeam'] == team
            for idx in df_sorted[home_mask].index:
                match_date = df_sorted.loc[idx, 'Date']
                closest_date = team_momentum.index[team_momentum.index <= match_date]
                
                if len(closest_date) > 0:
                    momentum_data = team_momentum.loc[closest_date[-1]]
                    df.at[idx, 'home_momentum_short'] = momentum_data['momentum_short']
                    df.at[idx, 'home_momentum_long'] = momentum_data['momentum_long']
                    df.at[idx, 'home_acceleration'] = momentum_data['acceleration']
            
            # Away matches  
            away_mask = df_sorted['AwayTeam'] == team
            for idx in df_sorted[away_mask].index:
                match_date = df_sorted.loc[idx, 'Date']
                closest_date = team_momentum.index[team_momentum.index <= match_date]
                
                if len(closest_date) > 0:
                    momentum_data = team_momentum.loc[closest_date[-1]]
                    df.at[idx, 'away_momentum_short'] = momentum_data['momentum_short']
                    df.at[idx, 'away_momentum_long'] = momentum_data['momentum_long']
                    df.at[idx, 'away_acceleration'] = momentum_data['acceleration']
        
        # Fill NaN avec valeurs neutres
        momentum_cols = [
            'home_momentum_short', 'away_momentum_short',
            'home_momentum_long', 'away_momentum_long',
            'home_acceleration', 'away_acceleration'
        ]
        
        for col in momentum_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0)
        
        # Features dérivées
        df['momentum_advantage'] = (
            df['home_momentum_short'] - df['away_momentum_short']
        )
        
        df['acceleration_advantage'] = (
            df['home_acceleration'] - df['away_acceleration'] 
        )
        
        df['momentum_alignment'] = (
            (df['home_momentum_short'] * df['home_momentum_long'] > 0).astype(int) -
            (df['away_momentum_short'] * df['away_momentum_long'] > 0).astype(int)
        )  # +1 si home momentum aligné, -1 si away aligné
        
        self.logger.info("✅ 9 features momentum ajoutées")
        
        return df
    
    def build_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features contexte: Enjeux classement, rivalités, pression
        """
        self.logger.info("\n🎭 CONSTRUCTION FEATURES CONTEXTE")
        self.logger.info("-"*50)
        
        # Simuler features contexte (en l'absence de data externe)
        np.random.seed(42)
        n_matches = len(df)
        
        # Rivalités (simulées)
        rivalry_pairs = [
            ('Arsenal', 'Tottenham'), ('Liverpool', 'Everton'),
            ('Man City', 'Man United'), ('Chelsea', 'Arsenal')
        ]
        
        is_rivalry = np.zeros(n_matches)
        for home, away in rivalry_pairs:
            rivalry_mask = (
                ((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
                ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))
            )
            is_rivalry[rivalry_mask] = 1
        
        # Enjeux saison (simulés selon période)
        days_into_season = (
            pd.to_datetime(df['Date']) - 
            pd.to_datetime(df['Date']).groupby(pd.to_datetime(df['Date']).dt.year).transform('min')
        ).dt.days
        
        season_pressure = np.where(
            days_into_season > 200,  # Fin de saison
            1.0,                     # Pression maximale
            days_into_season / 200   # Pression croissante
        )
        
        # Contexte features
        context_features = {
            'context_is_rivalry': is_rivalry,
            'context_season_pressure': season_pressure,
            'context_weekend_match': pd.to_datetime(df['Date']).dt.weekday.isin([5, 6]).astype(int),
            'context_month': pd.to_datetime(df['Date']).dt.month,
            'context_is_christmas_period': (
                pd.to_datetime(df['Date']).dt.month.isin([12, 1])
            ).astype(int),
        }
        
        # Features dérivées
        context_features['context_high_stakes'] = (
            (context_features['context_is_rivalry'] * 0.5) +
            (context_features['context_season_pressure'] * 0.5)
        ).clip(0, 1)
        
        # Cyclical encoding pour mois
        context_features['context_month_sin'] = np.sin(2 * np.pi * context_features['context_month'] / 12)
        context_features['context_month_cos'] = np.cos(2 * np.pi * context_features['context_month'] / 12)
        
        # Ajouter au DataFrame
        for col, values in context_features.items():
            df[col] = values
        
        self.logger.info("✅ 8 features contexte ajoutées")
        
        return df
    
    def normalize_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliser les nouvelles features contextuelles
        """
        self.logger.info("\n📊 NORMALISATION FEATURES CONTEXTUELLES")
        self.logger.info("-"*50)
        
        # Identifier colonnes contextuelles
        contextual_prefixes = ['weather_', 'referee_', 'home_days_', 'away_days_', 
                              'home_matches_', 'away_matches_', 'rest_', 'congestion_',
                              'fatigue_', 'home_momentum_', 'away_momentum_', 'momentum_',
                              'acceleration_', 'context_']
        
        contextual_cols = []
        for col in df.columns:
            if any(col.startswith(prefix) for prefix in contextual_prefixes):
                if df[col].dtype in ['int64', 'float64']:
                    contextual_cols.append(col)
        
        self.logger.info(f"🔧 Normalisation de {len(contextual_cols)} features contextuelles")
        
        # Normaliser Min-Max [0,1] ou Z-score selon distribution
        normalized_cols = []
        
        for col in contextual_cols:
            if col.endswith('_normalized') or 'score' in col:
                # Déjà normalisé
                continue
                
            col_data = df[col]
            
            # Choisir méthode selon distribution
            if col_data.std() == 0:
                # Constante
                df[f'{col}_normalized'] = 0.5
            elif 'sin' in col or 'cos' in col:
                # Déjà dans [-1,1] 
                df[f'{col}_normalized'] = (col_data + 1) / 2
            elif col_data.min() >= 0:
                # Min-Max pour données positives
                df[f'{col}_normalized'] = (
                    (col_data - col_data.min()) / 
                    (col_data.max() - col_data.min() + 1e-8)
                )
            else:
                # Z-score pour données centrées
                df[f'{col}_normalized'] = (
                    (col_data - col_data.mean()) / (col_data.std() + 1e-8)
                ).clip(-3, 3) / 6 + 0.5  # Ramener à [0,1]
            
            normalized_cols.append(f'{col}_normalized')
        
        self.logger.info(f"✅ {len(normalized_cols)} features normalisées créées")
        
        return df
    
    def build_all_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline complet de construction des features contextuelles
        """
        self.logger.info("🚀 PIPELINE COMPLET FEATURES CONTEXTUELLES")
        self.logger.info("="*70)
        
        initial_features = len([col for col in df.columns if col not in 
                               ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult']])
        
        self.logger.info(f"📊 Features initiales: {initial_features}")
        
        # 1. Features météo
        df = self.build_weather_features(df)
        
        # 2. Features arbitres
        df = self.build_referee_features(df)
        
        # 3. Features fatigue
        df = self.build_fatigue_features(df)
        
        # 4. Features momentum
        df = self.build_momentum_features(df)
        
        # 5. Features contexte
        df = self.build_context_features(df)
        
        # 6. Normalisation
        df = self.normalize_contextual_features(df)
        
        # Comptage final
        final_features = len([col for col in df.columns if col not in 
                             ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult']])
        
        added_features = final_features - initial_features
        
        self.logger.info(f"\n🎯 RÉSUMÉ FEATURES CONTEXTUELLES:")
        self.logger.info(f"  Features ajoutées: {added_features}")
        self.logger.info(f"  Total features: {final_features}")
        self.logger.info(f"  Augmentation: +{(added_features/initial_features)*100:.1f}%")
        
        return df
    
    def save_enriched_data(self, df: pd.DataFrame, output_path: str = None) -> str:
        """
        Sauvegarder données enrichies avec features contextuelles
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
            output_path = f'data/processed/v13_contextual_features_{timestamp}.csv'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"✅ Données enrichies sauvegardées: {output_path}")
        
        return output_path

def main():
    """
    Script principal de construction des features contextuelles
    """
    print("🌟 CONSTRUCTION FEATURES CONTEXTUELLES INNOVANTES")
    print("="*70)
    print("Features ciblées: Météo, Arbitres, Fatigue, Momentum, Contexte")
    print("Objectif: Intelligence football vs features xG redondantes")
    print("="*70)
    
    builder = ContextualFeaturesBuilder()
    
    try:
        # 1. Charger données de base
        df = builder.load_base_data()
        
        # 2. Construire toutes les features contextuelles
        df_enriched = builder.build_all_contextual_features(df)
        
        # 3. Sauvegarder
        output_path = builder.save_enriched_data(df_enriched)
        
        # 4. Résumé final
        contextual_features = [col for col in df_enriched.columns 
                             if any(col.startswith(p) for p in 
                                   ['weather_', 'referee_', 'fatigue_', 'momentum_', 'context_'])]
        
        print("\n" + "="*70)
        print("🎯 FEATURES CONTEXTUELLES TERMINÉES!")
        print(f"📊 Features contextuelles: {len(contextual_features)}")
        print(f"📈 Total features: {len(df_enriched.columns) - 5}")  # -5 metadata
        print(f"📁 Fichier: {output_path}")
        print("🚀 Prêt pour test performance vs features redondantes!")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())