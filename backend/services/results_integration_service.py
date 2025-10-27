"""
Results Integration Pipeline Service
Gère la récupération, validation et intégration des résultats réels des matchs EPL
pour comparer avec les prédictions et calculer les performances du modèle.
"""

import asyncio
import json
try:
    import aiohttp
    import ssl
    import certifi
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import os

from core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Structure pour un résultat de match validé"""
    match_id: str
    home_team: str
    away_team: str
    date: str
    home_score: int
    away_score: int
    result: str  # H/D/A
    status: str  # FT/postponed/cancelled
    source: str
    fetched_at: datetime
    gameweek: int

@dataclass
class PredictionAccuracy:
    """Métriques d'accuracy pour une prédiction"""
    match_id: str
    predicted: str
    actual: str
    correct: bool
    confidence: float
    brier_score: float
    market_beat: bool
    market_confidence: float

@dataclass
class GameweekPerformance:
    """Performance globale pour une gameweek"""
    gameweek: int
    total_matches: int
    correct_predictions: int
    accuracy: float
    avg_confidence: float
    avg_brier_score: float
    market_beat_rate: float
    best_prediction: Dict
    worst_prediction: Dict
    generated_at: datetime

class ResultsFetcher:
    """Service de récupération des résultats depuis multiples sources"""
    
    def __init__(self):
        self.session = None
        self.sources = {
            'football_data': {
                'url': 'https://api.football-data.org/v4/competitions/PL/matches',
                'headers': {'X-Auth-Token': settings.FOOTBALL_DATA_API_KEY},
                'priority': 1
            },
            'api_football': {
                'url': 'https://api.api-football.com/v3/fixtures',
                'headers': {'X-RapidAPI-Key': settings.API_FOOTBALL_KEY},
                'priority': 2
            },
            'openfootball': {
                'url': 'https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/en.1.json',
                'headers': {},
                'priority': 3
            }
        }
    
    async def __aenter__(self):
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for fetching results")
        
        # Créer un contexte SSL avec certificats
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10, ssl=ssl_context)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_gameweek_results(self, gameweek: int) -> List[MatchResult]:
        """Récupère les résultats pour une gameweek donnée"""
        logger.info(f"🔄 Fetching results for GW{gameweek}")
        
        all_results = []
        
        # Essayer chaque source par ordre de priorité
        for source_name, config in sorted(self.sources.items(), key=lambda x: x[1]['priority']):
            try:
                results = await self._fetch_from_source(source_name, config, gameweek)
                if results:
                    logger.info(f"✅ Got {len(results)} results from {source_name}")
                    all_results.extend(results)
                    break  # Première source qui marche
            except Exception as e:
                logger.warning(f"⚠️ {source_name} failed: {e}")
                continue
        
        return self._deduplicate_results(all_results)
    
    async def _fetch_from_source(self, source_name: str, config: Dict, gameweek: int) -> List[MatchResult]:
        """Récupère depuis une source spécifique"""
        if source_name == 'football_data':
            return await self._fetch_football_data(config, gameweek)
        elif source_name == 'api_football':
            return await self._fetch_api_football(config, gameweek)
        elif source_name == 'openfootball':
            return await self._fetch_openfootball(config, gameweek)
        
        return []
    
    async def _fetch_football_data(self, config: Dict, gameweek: int) -> List[MatchResult]:
        """Récupère depuis Football-Data.org"""
        url = f"{config['url']}?matchday={gameweek}&status=FINISHED"
        
        async with self.session.get(url, headers=config['headers']) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}")
            
            data = await response.json()
            results = []
            
            for match in data.get('matches', []):
                if match['status'] == 'FINISHED':
                    home_score = match['score']['fullTime']['home']
                    away_score = match['score']['fullTime']['away']
                    
                    if home_score > away_score:
                        result = 'H'
                    elif home_score < away_score:
                        result = 'A'
                    else:
                        result = 'D'
                    
                    home_normalized = self._normalize_team_name(match['homeTeam']['name'])
                    away_normalized = self._normalize_team_name(match['awayTeam']['name'])
                    logger.info(f"🏠 {match['homeTeam']['name']} -> {home_normalized}")
                    logger.info(f"🛣️ {match['awayTeam']['name']} -> {away_normalized}")
                    
                    results.append(MatchResult(
                        match_id=f"fd_{match['id']}",
                        home_team=home_normalized,
                        away_team=away_normalized,
                        date=match['utcDate'][:10],
                        home_score=home_score,
                        away_score=away_score,
                        result=result,
                        status='FT',
                        source='football_data',
                        fetched_at=datetime.now(),
                        gameweek=gameweek
                    ))
            
            return results
    
    async def _fetch_api_football(self, config: Dict, gameweek: int) -> List[MatchResult]:
        """Récupère depuis API-Football"""
        # TODO: Implémenter selon leur format
        return []
    
    async def _fetch_openfootball(self, config: Dict, gameweek: int) -> List[MatchResult]:
        """Récupère depuis OpenFootball JSON"""
        # TODO: Implémenter selon leur format  
        return []
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalise les noms d'équipes pour matcher nos prédictions"""
        mapping = {
            # Football-Data API -> Our prediction format
            'Manchester United FC': 'Man_Utd',
            'Manchester City FC': 'Man_City', 
            'Tottenham Hotspur FC': 'Tottenham',
            'Brighton & Hove Albion FC': 'Brighton',
            'Newcastle United FC': 'Newcastle',
            'Nottingham Forest FC': 'Nottingham_Forest',
            'West Ham United FC': 'West_Ham',
            'Aston Villa FC': 'Aston_Villa',
            'Crystal Palace FC': 'Crystal_Palace',
            'Arsenal FC': 'Arsenal',
            'Chelsea FC': 'Chelsea',
            'Liverpool FC': 'Liverpool',
            'Brentford FC': 'Brentford',
            'AFC Bournemouth': 'Bournemouth',
            'Wolverhampton Wanderers FC': 'Wolves',
            'Burnley FC': 'Burnley',
            'Everton FC': 'Everton',
            'Fulham FC': 'Fulham',
            'Leeds United FC': 'Leeds',
            'Sunderland AFC': 'Sunderland',
            # Common shortenings from API
            'Man Utd': 'Man_Utd',
            'Man City': 'Man_City',
            'Spurs': 'Tottenham',
            "Nott'm Forest": 'Nottingham_Forest'
        }
        
        return mapping.get(team_name, team_name)
    
    def _deduplicate_results(self, results: List[MatchResult]) -> List[MatchResult]:
        """Déduplique les résultats par match"""
        seen = set()
        unique_results = []
        
        for result in results:
            match_key = f"{result.home_team}_vs_{result.away_team}_{result.date}"
            if match_key not in seen:
                seen.add(match_key)
                unique_results.append(result)
        
        return unique_results

class AccuracyCalculator:
    """Calculateur de métriques d'accuracy et performance"""
    
    @staticmethod
    def calculate_brier_score(predicted_probs: Dict[str, float], actual_result: str) -> float:
        """Calcule le Brier Score (plus bas = mieux)"""
        brier = 0.0
        for outcome in ['home', 'draw', 'away']:
            predicted = predicted_probs.get(outcome, 0.0)
            actual = 1.0 if (outcome == 'home' and actual_result == 'H') or \
                           (outcome == 'draw' and actual_result == 'D') or \
                           (outcome == 'away' and actual_result == 'A') else 0.0
            brier += (predicted - actual) ** 2
        
        return brier / 3  # Normaliser par nombre d'outcomes
    
    @staticmethod
    def beats_market(predicted_probs: Dict[str, float], market_probs: Dict[str, float], 
                    actual_result: str) -> bool:
        """Vérifie si notre prédiction bat le marché"""
        our_confidence = max(predicted_probs.values())
        market_confidence = max(market_probs.values()) if market_probs else 0.0
        
        # Notre prédiction bat le marché si elle est plus confiante ET correcte
        our_prediction = max(predicted_probs, key=predicted_probs.get)
        our_letter = {'home': 'H', 'draw': 'D', 'away': 'A'}[our_prediction]
        
        return our_letter == actual_result and our_confidence > market_confidence

class ResultsIntegrationService:
    """Service principal d'intégration des résultats"""
    
    def __init__(self):
        self.fetcher = ResultsFetcher()
        self.calculator = AccuracyCalculator()
        self.results_dir = Path(settings.DATA_DIR) / "results"
        self.results_dir.mkdir(exist_ok=True)
        
    def discover_available_gameweeks(self) -> List[int]:
        """Découvre automatiquement les gameweeks avec prédictions disponibles"""
        predictions_dir = Path(settings.PIPELINE_PREDICTIONS_DIR)
        if not predictions_dir.exists():
            return []
            
        gameweeks = set()
        
        # Scanner les dossiers gw{X} et {X}
        for item in predictions_dir.iterdir():
            if item.is_dir():
                # Extraire le numéro de gameweek
                if item.name.startswith('gw') and item.name[2:].isdigit():
                    gw_num = int(item.name[2:])
                    if 1 <= gw_num <= 38:
                        gameweeks.add(gw_num)
                elif item.name.isdigit():
                    gw_num = int(item.name)
                    if 1 <= gw_num <= 38:
                        gameweeks.add(gw_num)
        
        return sorted(list(gameweeks))
    
    async def auto_process_all_available_gameweeks(self):
        """Traite automatiquement toutes les gameweeks disponibles"""
        available_gws = self.discover_available_gameweeks()
        logger.info(f"🔍 Found {len(available_gws)} gameweeks with predictions: {available_gws}")
        
        for gw in available_gws:
            try:
                # Vérifier si déjà traité
                performance_file = self.results_dir / f"gw{gw}" / "performance.json"
                if performance_file.exists():
                    logger.info(f"✅ GW{gw} already processed, skipping")
                    continue
                    
                logger.info(f"🔄 Processing GW{gw}...")
                await self.process_gameweek_results(gw)
                
            except Exception as e:
                logger.warning(f"⚠️ Could not process GW{gw}: {e}")
                continue
        
    async def process_gameweek_results(self, gameweek: int) -> GameweekPerformance:
        """Traite les résultats d'une gameweek complète"""
        logger.info(f"📊 Processing results for GW{gameweek}")
        
        # 1. Récupérer les résultats réels
        async with self.fetcher as fetcher:
            results = await fetcher.fetch_gameweek_results(gameweek)
        
        if not results:
            raise Exception(f"No results found for GW{gameweek}")
        
        # 2. Charger nos prédictions
        predictions = await self._load_predictions(gameweek)
        
        # 3. Calculer les métriques
        accuracies = []
        matches_data = predictions.get('matches', predictions)  # Support both formats
        
        for result in results:
            match_key = f"{result.home_team}_vs_{result.away_team}"
            logger.info(f"🔍 Looking for match: {match_key}")
            
            if match_key in matches_data:
                pred = matches_data[match_key]
                accuracy = self._calculate_match_accuracy(pred, result)
                accuracies.append(accuracy)
                logger.info(f"✅ Found prediction for {match_key}")
            else:
                logger.warning(f"⚠️ No prediction found for {match_key}")
                logger.info(f"🔍 Available predictions: {list(matches_data.keys())}")
        
        # 4. Agréger les performances
        performance = self._aggregate_performance(gameweek, accuracies)
        
        # 5. Sauvegarder
        await self._save_results(gameweek, results, accuracies, performance)
        
        logger.info(f"✅ GW{gameweek} processed: {performance.accuracy:.1%} accuracy")
        return performance
    
    async def _load_predictions(self, gameweek: int) -> Dict:
        """Charge nos prédictions pour une gameweek"""
        # Chercher le fichier de prédiction le plus récent pour cette GW
        predictions_dir = Path(settings.PIPELINE_PREDICTIONS_DIR)
        
        # Patterns possibles: gw{X}/, {X}/, etc.
        possible_paths = [
            predictions_dir / f"gw{gameweek}" / "predictions.json",
            predictions_dir / f"{gameweek}" / "predictions.json", 
            predictions_dir / f"gw{gameweek}" / "prediction.json",
            predictions_dir / f"{gameweek}" / "prediction.json"
        ]
        
        pred_file = None
        for path in possible_paths:
            if path.exists():
                pred_file = path
                break
        
        if not pred_file:
            raise Exception(f"No predictions found for GW{gameweek}")
        
        with open(pred_file) as f:
            data = json.load(f)
            # Support différents formats
            predictions = data.get('predictions', data.get('matches', {}))
            return predictions
    
    def _calculate_match_accuracy(self, prediction: Dict, result: MatchResult) -> PredictionAccuracy:
        """Calcule les métriques pour un match"""
        predicted = prediction['prediction']
        actual = result.result
        correct = predicted == actual
        
        confidence = prediction['confidence']
        probs = prediction['probabilities']
        
        brier_score = self.calculator.calculate_brier_score(probs, actual)
        
        market_probs = prediction.get('market_features', {})
        market_confidence = market_probs.get('market_confidence', 0.0)
        market_beat = self.calculator.beats_market(probs, {}, actual)
        
        return PredictionAccuracy(
            match_id=f"{result.home_team}_vs_{result.away_team}",
            predicted=predicted,
            actual=actual,
            correct=correct,
            confidence=confidence,
            brier_score=brier_score,
            market_beat=market_beat,
            market_confidence=market_confidence
        )
    
    def _aggregate_performance(self, gameweek: int, accuracies: List[PredictionAccuracy]) -> GameweekPerformance:
        """Agrège les performances d'une gameweek"""
        total = len(accuracies)
        correct = sum(1 for acc in accuracies if acc.correct)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_confidence = sum(acc.confidence for acc in accuracies) / total if total > 0 else 0.0
        avg_brier = sum(acc.brier_score for acc in accuracies) / total if total > 0 else 0.0
        market_beats = sum(1 for acc in accuracies if acc.market_beat)
        market_beat_rate = market_beats / total if total > 0 else 0.0
        
        # Meilleure et pire prédiction
        best = max(accuracies, key=lambda x: x.confidence if x.correct else 0, default=None)
        worst = min(accuracies, key=lambda x: x.confidence if not x.correct else 1, default=None)
        
        return GameweekPerformance(
            gameweek=gameweek,
            total_matches=total,
            correct_predictions=correct,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            avg_brier_score=avg_brier,
            market_beat_rate=market_beat_rate,
            best_prediction=asdict(best) if best else {},
            worst_prediction=asdict(worst) if worst else {},
            generated_at=datetime.now()
        )
    
    async def _save_results(self, gameweek: int, results: List[MatchResult], 
                          accuracies: List[PredictionAccuracy], performance: GameweekPerformance):
        """Sauvegarde les résultats et métriques"""
        gw_dir = self.results_dir / f"gw{gameweek}"
        gw_dir.mkdir(exist_ok=True)
        
        # Résultats bruts
        with open(gw_dir / "results.json", 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        
        # Métriques détaillées
        with open(gw_dir / "accuracies.json", 'w') as f:
            json.dump([asdict(a) for a in accuracies], f, indent=2, default=str)
        
        # Performance globale
        with open(gw_dir / "performance.json", 'w') as f:
            json.dump(asdict(performance), f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {gw_dir}")

# Service singleton
results_service = ResultsIntegrationService()