"""
API endpoints pour l'intégration et l'analyse des résultats
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from services.results_integration_service import results_service, GameweekPerformance
from services.real_time_polling_service import polling_manager
from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/results", tags=["results"])

@router.post("/process/{gameweek}")
async def process_gameweek_results(
    gameweek: int,
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Force reprocessing even if already done")
):
    """
    Déclenche le traitement des résultats pour une gameweek
    """
    try:
        if not (1 <= gameweek <= 38):
            raise HTTPException(status_code=400, detail="Gameweek must be between 1 and 38")
        
        logger.info(f"🎯 Processing results for GW{gameweek} (force={force})")
        
        # Traitement en arrière-plan
        background_tasks.add_task(
            _process_gameweek_background,
            gameweek,
            force
        )
        
        return {
            "message": f"Started processing GW{gameweek} results",
            "gameweek": gameweek,
            "force": force,
            "status": "processing"
        }
    
    except Exception as e:
        logger.error(f"Error starting GW{gameweek} processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_gameweek_background(gameweek: int, force: bool):
    """Traitement en arrière-plan d'une gameweek"""
    try:
        performance = await results_service.process_gameweek_results(gameweek)
        logger.info(f"✅ GW{gameweek} processed successfully: {performance.accuracy:.1%} accuracy")
    except Exception as e:
        logger.error(f"❌ Failed to process GW{gameweek}: {e}")

@router.post("/auto-process")
async def auto_process_all_gameweeks(background_tasks: BackgroundTasks):
    """
    Traite automatiquement toutes les gameweeks disponibles
    """
    try:
        logger.info("🚀 Starting auto-processing of all available gameweeks")
        
        # Traitement en arrière-plan
        background_tasks.add_task(results_service.auto_process_all_available_gameweeks)
        
        return {
            "message": "Started auto-processing all available gameweeks",
            "status": "processing"
        }
    
    except Exception as e:
        logger.error(f"Error starting auto-processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/overview")
async def get_performance_overview(
    start_gw: Optional[int] = Query(None, description="Start gameweek"),
    end_gw: Optional[int] = Query(None, description="End gameweek"),
    limit: int = Query(10, description="Number of recent gameweeks to include")
) -> Dict[str, Any]:
    """
    Récupère un aperçu des performances sur plusieurs gameweeks
    """
    try:
        results_dir = settings.DATA_DIR / "results"
        
        if not results_dir.exists():
            return {
                "gameweeks": [],
                "overall_stats": {
                    "total_games": 0,
                    "overall_accuracy": 0.0,
                    "avg_confidence": 0.0,
                    "market_beat_rate": 0.0
                },
                "message": "No performance data available yet"
            }
        
        # Lister les gameweeks disponibles
        available_gws = []
        for gw_dir in results_dir.iterdir():
            if gw_dir.is_dir() and gw_dir.name.startswith("gw"):
                try:
                    gw_num = int(gw_dir.name[2:])
                    performance_file = gw_dir / "performance.json"
                    if performance_file.exists():
                        available_gws.append(gw_num)
                except ValueError:
                    continue
        
        available_gws.sort()
        
        # Filtrer selon les paramètres
        if start_gw:
            available_gws = [gw for gw in available_gws if gw >= start_gw]
        if end_gw:
            available_gws = [gw for gw in available_gws if gw <= end_gw]
        
        # Limiter le nombre
        available_gws = available_gws[-limit:]
        
        # Charger les données de performance
        gameweeks_data = []
        total_games = 0
        total_correct = 0
        total_confidence = 0.0
        total_market_beats = 0
        
        import json
        for gw in available_gws:
            try:
                performance_file = results_dir / f"gw{gw}" / "performance.json"
                with open(performance_file) as f:
                    data = json.load(f)
                    gameweeks_data.append(data)
                    
                    # Agréger pour les stats globales
                    total_games += data.get("total_matches", 0)
                    total_correct += data.get("correct_predictions", 0)
                    total_confidence += data.get("avg_confidence", 0.0)
                    total_market_beats += data.get("market_beat_rate", 0.0) * data.get("total_matches", 0)
            
            except Exception as e:
                logger.warning(f"Could not load GW{gw} performance: {e}")
                continue
        
        # Calculer les stats globales
        num_gws = len(gameweeks_data)
        overall_stats = {
            "total_games": total_games,
            "overall_accuracy": total_correct / total_games if total_games > 0 else 0.0,
            "avg_confidence": total_confidence / num_gws if num_gws > 0 else 0.0,
            "market_beat_rate": total_market_beats / total_games if total_games > 0 else 0.0,
            "gameweeks_analyzed": num_gws,
            "best_gameweek": max(gameweeks_data, key=lambda x: x.get("accuracy", 0), default={}).get("gameweek"),
            "best_accuracy": max(gw.get("accuracy", 0) for gw in gameweeks_data) if gameweeks_data else 0.0
        }
        
        return {
            "gameweeks": gameweeks_data,
            "overall_stats": overall_stats,
            "period": {
                "start_gw": min(available_gws) if available_gws else None,
                "end_gw": max(available_gws) if available_gws else None,
                "total_gameweeks": len(available_gws)
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting performance overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/{gameweek}")
async def get_gameweek_performance(gameweek: int) -> Dict[str, Any]:
    """
    Récupère les performances d'une gameweek
    """
    try:
        if not (1 <= gameweek <= 38):
            raise HTTPException(status_code=400, detail="Gameweek must be between 1 and 38")
        
        # Charger depuis le fichier de performance
        performance_file = settings.DATA_DIR / "results" / f"gw{gameweek}" / "performance.json"
        
        if not performance_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"No performance data found for GW{gameweek}"
            )
        
        import json
        with open(performance_file) as f:
            data = json.load(f)
        
        return {
            "gameweek": gameweek,
            "performance": data,
            "available": True
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading GW{gameweek} performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/polling/start/{gameweek}")
async def start_polling_gameweek(gameweek: int):
    """
    Démarre la surveillance automatique d'une gameweek
    """
    try:
        if not (1 <= gameweek <= 38):
            raise HTTPException(status_code=400, detail="Gameweek must be between 1 and 38")
        
        await polling_manager.monitor_gameweek(gameweek)
        
        return {
            "message": f"Started monitoring GW{gameweek}",
            "gameweek": gameweek,
            "status": "monitoring"
        }
    
    except Exception as e:
        logger.error(f"Error starting GW{gameweek} monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/polling/stop/{gameweek}")
async def stop_polling_gameweek(gameweek: int):
    """
    Arrête la surveillance d'une gameweek
    """
    try:
        await polling_manager.stop_monitoring(gameweek)
        
        return {
            "message": f"Stopped monitoring GW{gameweek}",
            "gameweek": gameweek,
            "status": "stopped"
        }
    
    except Exception as e:
        logger.error(f"Error stopping GW{gameweek} monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/polling/status")
async def get_polling_status():
    """
    Récupère le statut de la surveillance
    """
    try:
        status = polling_manager.get_monitoring_status()
        
        return {
            "polling_status": status,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting polling status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/matches/{gameweek}")
async def get_gameweek_results(gameweek: int):
    """
    Récupère les résultats bruts d'une gameweek
    """
    try:
        if not (1 <= gameweek <= 38):
            raise HTTPException(status_code=400, detail="Gameweek must be between 1 and 38")
        
        results_file = settings.DATA_DIR / "results" / f"gw{gameweek}" / "results.json"
        
        if not results_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No results found for GW{gameweek}"
            )
        
        import json
        with open(results_file) as f:
            results = json.load(f)
        
        return {
            "gameweek": gameweek,
            "results": results,
            "count": len(results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading GW{gameweek} results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/accuracy/{gameweek}")
async def get_match_accuracies(gameweek: int):
    """
    Récupère les métriques détaillées de précision pour une gameweek
    """
    try:
        if not (1 <= gameweek <= 38):
            raise HTTPException(status_code=400, detail="Gameweek must be between 1 and 38")
        
        accuracies_file = settings.DATA_DIR / "results" / f"gw{gameweek}" / "accuracies.json"
        
        if not accuracies_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No accuracy data found for GW{gameweek}"
            )
        
        import json
        with open(accuracies_file) as f:
            accuracies = json.load(f)
        
        return {
            "gameweek": gameweek,
            "accuracies": accuracies,
            "count": len(accuracies)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading GW{gameweek} accuracies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/performance/{gameweek}")
async def clear_gameweek_data(gameweek: int, confirm: bool = Query(False)):
    """
    Supprime les données de performance d'une gameweek
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="Must set confirm=true to delete data"
            )
        
        if not (1 <= gameweek <= 38):
            raise HTTPException(status_code=400, detail="Gameweek must be between 1 and 38")
        
        gw_dir = settings.DATA_DIR / "results" / f"gw{gameweek}"
        
        if not gw_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No data found for GW{gameweek}"
            )
        
        import shutil
        shutil.rmtree(gw_dir)
        
        logger.info(f"🗑️ Cleared GW{gameweek} performance data")
        
        return {
            "message": f"Cleared GW{gameweek} data",
            "gameweek": gameweek,
            "status": "deleted"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing GW{gameweek} data: {e}")
        raise HTTPException(status_code=500, detail=str(e))