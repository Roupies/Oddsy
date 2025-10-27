"""
Real-Time Polling Service pour la détection automatique des résultats
Surveille en continu les matches en cours et déclenche l'intégration des résultats
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from .results_integration_service import results_service, ResultsFetcher, MatchResult
from core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class MatchStatus:
    """Statut d'un match en cours de surveillance"""
    match_id: str
    home_team: str
    away_team: str
    kickoff_time: datetime
    gameweek: int
    status: str  # scheduled/live/finished/processed
    last_check: datetime
    result: Optional[MatchResult] = None

@dataclass
class PollingConfig:
    """Configuration du polling"""
    check_interval_scheduled: int = 300  # 5 minutes pour matches programmés
    check_interval_live: int = 60        # 1 minute pour matches en cours
    check_interval_finished: int = 900   # 15 minutes pour matches finis
    max_retries: int = 3
    retry_delay: int = 30

class MatchScheduler:
    """Gestionnaire de planning des matches"""
    
    def __init__(self):
        self.matches_file = Path(settings.DATA_DIR) / "scheduling" / "matches.json"
        self.matches_file.parent.mkdir(exist_ok=True)
    
    async def load_gameweek_schedule(self, gameweek: int) -> List[MatchStatus]:
        """Charge le planning d'une gameweek"""
        try:
            # Charger depuis nos prédictions d'abord
            pred_file = Path(f"prediction/{gameweek}/prediction.json")
            if not pred_file.exists():
                logger.warning(f"No predictions file for GW{gameweek}")
                return []
            
            with open(pred_file) as f:
                data = json.load(f)
            
            matches = []
            for match_key, prediction in data.get('predictions', {}).items():
                match_info = prediction.get('match_info', {})
                
                # Parser la date/heure du match
                match_date = match_info.get('date', '')
                kickoff_time = self._parse_match_time(match_date)
                
                matches.append(MatchStatus(
                    match_id=match_key,
                    home_team=match_info.get('home', ''),
                    away_team=match_info.get('away', ''),
                    kickoff_time=kickoff_time,
                    gameweek=gameweek,
                    status='scheduled',
                    last_check=datetime.now()
                ))
            
            logger.info(f"📅 Loaded {len(matches)} matches for GW{gameweek}")
            return matches
        
        except Exception as e:
            logger.error(f"Error loading GW{gameweek} schedule: {e}")
            return []
    
    def _parse_match_time(self, date_str: str) -> datetime:
        """Parse la date/heure d'un match"""
        try:
            # Format basique: "2025-10-24"
            if len(date_str) == 10:
                # Ajouter heure par défaut (15h pour Premier League)
                return datetime.strptime(f"{date_str} 15:00", "%Y-%m-%d %H:%M")
            else:
                # Format avec heure incluse
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            # Fallback sur maintenant + 1h
            return datetime.now() + timedelta(hours=1)

class RealTimePoller:
    """Service de polling temps réel"""
    
    def __init__(self):
        self.config = PollingConfig()
        self.scheduler = MatchScheduler()
        self.fetcher = ResultsFetcher()
        self.active_gameweeks: Set[int] = set()
        self.polling_tasks: Dict[int, asyncio.Task] = {}
        self.notification_callbacks: List = []
    
    def add_notification_callback(self, callback):
        """Ajoute un callback pour les notifications"""
        self.notification_callbacks.append(callback)
    
    async def start_monitoring_gameweek(self, gameweek: int):
        """Démarre la surveillance d'une gameweek"""
        if gameweek in self.active_gameweeks:
            logger.info(f"GW{gameweek} already being monitored")
            return
        
        logger.info(f"🎯 Starting monitoring for GW{gameweek}")
        self.active_gameweeks.add(gameweek)
        
        # Créer la tâche de polling
        task = asyncio.create_task(self._poll_gameweek(gameweek))
        self.polling_tasks[gameweek] = task
        
        await self._notify(f"Started monitoring GW{gameweek}")
    
    async def stop_monitoring_gameweek(self, gameweek: int):
        """Arrête la surveillance d'une gameweek"""
        if gameweek not in self.active_gameweeks:
            return
        
        logger.info(f"🛑 Stopping monitoring for GW{gameweek}")
        
        # Annuler la tâche
        if gameweek in self.polling_tasks:
            self.polling_tasks[gameweek].cancel()
            del self.polling_tasks[gameweek]
        
        self.active_gameweeks.remove(gameweek)
        await self._notify(f"Stopped monitoring GW{gameweek}")
    
    async def _poll_gameweek(self, gameweek: int):
        """Boucle de polling pour une gameweek"""
        matches = await self.scheduler.load_gameweek_schedule(gameweek)
        
        if not matches:
            logger.warning(f"No matches to monitor for GW{gameweek}")
            return
        
        logger.info(f"🔄 Polling {len(matches)} matches for GW{gameweek}")
        
        try:
            while gameweek in self.active_gameweeks:
                await self._check_matches(matches)
                
                # Vérifier si toutes les matches sont terminées et traitées
                if all(m.status == 'processed' for m in matches):
                    logger.info(f"✅ All matches processed for GW{gameweek}")
                    await self.stop_monitoring_gameweek(gameweek)
                    break
                
                # Attendre avant la prochaine vérification
                await asyncio.sleep(self._get_check_interval(matches))
        
        except asyncio.CancelledError:
            logger.info(f"Polling cancelled for GW{gameweek}")
        except Exception as e:
            logger.error(f"Error in polling GW{gameweek}: {e}")
            await self._notify(f"Error monitoring GW{gameweek}: {e}")
    
    async def _check_matches(self, matches: List[MatchStatus]):
        """Vérifie le statut de tous les matches"""
        now = datetime.now()
        
        for match in matches:
            if match.status == 'processed':
                continue
            
            # Déterminer le statut du match basé sur l'heure
            if now < match.kickoff_time - timedelta(minutes=30):
                new_status = 'scheduled'
            elif now < match.kickoff_time + timedelta(minutes=120):  # 2h après coup d'envoi
                new_status = 'live'
            else:
                new_status = 'finished'
            
            # Mettre à jour le statut si changé
            if match.status != new_status:
                logger.info(f"📊 {match.match_id}: {match.status} → {new_status}")
                match.status = new_status
                await self._notify(f"Match {match.match_id} is now {new_status}")
            
            # Vérifier les résultats pour les matches finis
            if match.status == 'finished' and not match.result:
                await self._check_match_result(match)
    
    async def _check_match_result(self, match: MatchStatus):
        """Vérifie si un résultat est disponible pour un match"""
        try:
            async with self.fetcher as fetcher:
                results = await fetcher.fetch_gameweek_results(match.gameweek)
            
            # Chercher le résultat pour ce match
            for result in results:
                if (result.home_team == match.home_team and 
                    result.away_team == match.away_team):
                    
                    logger.info(f"🏆 Result found for {match.match_id}: {result.home_score}-{result.away_score} ({result.result})")
                    match.result = result
                    
                    # Déclencher le traitement des performances
                    await self._process_match_performance(match)
                    break
        
        except Exception as e:
            logger.warning(f"Failed to fetch result for {match.match_id}: {e}")
    
    async def _process_match_performance(self, match: MatchStatus):
        """Traite les performances d'un match individuel"""
        try:
            # Ici on pourrait traiter match par match ou attendre la gameweek complète
            logger.info(f"⚡ Processing performance for {match.match_id}")
            
            # Marquer comme traité
            match.status = 'processed'
            
            await self._notify(f"Performance calculated for {match.match_id}")
            
            # Vérifier si toute la gameweek est terminée
            await self._check_gameweek_completion(match.gameweek)
        
        except Exception as e:
            logger.error(f"Error processing performance for {match.match_id}: {e}")
    
    async def _check_gameweek_completion(self, gameweek: int):
        """Vérifie si une gameweek est complètement terminée"""
        try:
            # Déclencher le traitement complet de la gameweek
            performance = await results_service.process_gameweek_results(gameweek)
            
            logger.info(f"🎯 GW{gameweek} complete! Accuracy: {performance.accuracy:.1%}")
            await self._notify(f"GW{gameweek} performance: {performance.accuracy:.1%} accuracy")
        
        except Exception as e:
            logger.warning(f"Could not process complete GW{gameweek}: {e}")
    
    def _get_check_interval(self, matches: List[MatchStatus]) -> int:
        """Détermine l'intervalle de vérification basé sur les statuts"""
        statuses = [m.status for m in matches]
        
        if 'live' in statuses:
            return self.config.check_interval_live
        elif 'finished' in statuses:
            return self.config.check_interval_finished
        else:
            return self.config.check_interval_scheduled
    
    async def _notify(self, message: str):
        """Envoie une notification"""
        logger.info(f"📢 {message}")
        
        for callback in self.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

class PollingManager:
    """Gestionnaire principal du polling"""
    
    def __init__(self):
        self.poller = RealTimePoller()
        self.auto_monitoring = True
    
    async def start(self):
        """Démarre le service de polling"""
        logger.info("🚀 Starting Real-Time Polling Service")
        
        if self.auto_monitoring:
            # Démarrer automatiquement la surveillance des gameweeks récentes
            current_gw = self._get_current_gameweek()
            await self.poller.start_monitoring_gameweek(current_gw)
    
    async def stop(self):
        """Arrête le service de polling"""
        logger.info("🛑 Stopping Real-Time Polling Service")
        
        # Arrêter toutes les surveillances
        for gameweek in list(self.poller.active_gameweeks):
            await self.poller.stop_monitoring_gameweek(gameweek)
    
    def _get_current_gameweek(self) -> int:
        """Détermine la gameweek courante"""
        # Logique simple basée sur la date
        # TODO: Améliorer avec calendrier EPL réel
        season_start = datetime(2025, 8, 16)  # Début saison 2025-26
        weeks_since_start = (datetime.now() - season_start).days // 7
        return min(max(weeks_since_start + 1, 1), 38)
    
    async def monitor_gameweek(self, gameweek: int):
        """API pour démarrer la surveillance d'une gameweek"""
        await self.poller.start_monitoring_gameweek(gameweek)
    
    async def stop_monitoring(self, gameweek: int):
        """API pour arrêter la surveillance d'une gameweek"""
        await self.poller.stop_monitoring_gameweek(gameweek)
    
    def get_monitoring_status(self) -> Dict:
        """Retourne le statut de la surveillance"""
        return {
            'active_gameweeks': list(self.poller.active_gameweeks),
            'polling_tasks': len(self.poller.polling_tasks),
            'auto_monitoring': self.auto_monitoring
        }

# Service singleton
polling_manager = PollingManager()