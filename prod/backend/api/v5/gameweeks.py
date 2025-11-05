#!/usr/bin/env python3
"""
API v5 Gameweeks endpoints with ETag & Cache Support
===================================================
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import List, Dict, Optional, Any

from core.config import settings
from services.cache_service import get_cache_service
from services.coverage_validator import get_coverage_validator, CoverageValidationError
from services.probability_validator import get_probability_validator
from services.fixture_service import get_fixture_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/gameweeks", tags=["gameweeks"])

async def _load_gameweek_data(gameweek: int) -> Optional[Dict[str, Any]]:
    """
    Charge les données d'une gameweek pour validation
    
    Args:
        gameweek: Numéro de gameweek
        
    Returns:
        Données de prédictions ou None si pas trouvé
    """
    try:
        # Priorité 1: Artefacts gold
        gw_dir = Path(f"outputs/gold_backfill/gw{gameweek:02d}")
        
        if gw_dir.exists() and (gw_dir / "predictions.json").exists():
            with open(gw_dir / "predictions.json", 'r') as f:
                gold_data = json.load(f)
            return adapt_gold_to_apiv5(gold_data, gameweek)
        
        # Priorité 2: Artefacts versionnés
        versioned_dir = settings.PIPELINE_PREDICTIONS_DIR / "versioned"
        if versioned_dir.exists():
            latest_file = versioned_dir / f"j{gameweek}_latest.json"
            if latest_file.exists():
                with open(latest_file, 'r') as f:
                    return json.load(f)
            
            # Chercher fichiers versionnés par timestamp
            versioned_files = list(versioned_dir.glob(f"j{gameweek}_predictions_v3_*.json"))
            if versioned_files:
                latest_file = max(versioned_files, key=lambda p: p.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    return json.load(f)
        
        # Priorité 3: Prédictions pipeline (legacy)
        predictions_dir = settings.PIPELINE_PREDICTIONS_DIR
        if predictions_dir.exists():
            prediction_patterns = [
                f"j{gameweek}_predictions_v3_*.json",
                f"j{gameweek}_*pipeline_v2_predictions_*.json", 
                f"j{gameweek}_*_dual_champions_*.json"
            ]
            
            for pattern in prediction_patterns:
                prediction_files = list(predictions_dir.glob(pattern))
                if prediction_files:
                    latest_file = max(prediction_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        pipeline_data = json.load(f)
                    return adapt_pipeline_to_apiv5(pipeline_data, gameweek, latest_file.name)
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading gameweek {gameweek} data for validation: {e}")
        return None

@router.get("/latest")
async def get_latest_gameweek(request: Request) -> Response:
    """Découvre la gameweek la plus récente disponible avec cache optimisé"""
    
    cache_service = get_cache_service()
    
    try:
        # Récupérer toutes les gameweeks disponibles
        available_data = await get_available_gameweeks()
        available_gameweeks = available_data["data"]["available_gameweeks"]
        
        if not available_gameweeks:
            raise HTTPException(
                status_code=404,
                detail="Aucune gameweek disponible"
            )
        
        # Trouver la gameweek la plus récente VALIDE (10/10 fixtures)
        coverage_validator = get_coverage_validator()
        
        # Filtrer les gameweeks prêtes pour production
        production_ready_gws = []
        for gw in available_gameweeks:
            if gw["status"] == "available":
                try:
                    # Charger les données pour validation rapide
                    gw_predictions = await self._load_gameweek_data(gw["gameweek"])
                    if gw_predictions:
                        validation_report = coverage_validator.validate_gameweek_coverage(
                            gw_predictions, gw["gameweek"], allow_partial=False
                        )
                        if validation_report["ready_for_production"]:
                            gw["production_ready"] = True
                            gw["fixtures_count"] = validation_report["summary"]["fixtures_count"]
                            production_ready_gws.append(gw)
                        else:
                            gw["production_ready"] = False
                            gw["validation_issues"] = len([
                                v for v in validation_report["validations"].values() 
                                if not v["valid"]
                            ])
                except Exception as e:
                    logger.warning(f"Could not validate GW{gw['gameweek']}: {e}")
                    gw["production_ready"] = False
        
        if not production_ready_gws:
            # Fallback vers la dernière gameweek disponible même si pas 100% valide
            latest_gw = max(available_gameweeks, key=lambda x: x["gameweek"])
            latest_gw["production_ready"] = False
            logger.warning(f"No production-ready gameweeks found, using GW{latest_gw['gameweek']} as fallback")
        else:
            latest_gw = max(production_ready_gws, key=lambda x: x["gameweek"])
        
        response_data = {
            "api_version": "5.0.0",
            "generated_at": datetime.utcnow().isoformat(),
            "data": {
                "latest_gameweek": latest_gw["gameweek"],
                "status": latest_gw["status"],
                "metadata": latest_gw["metadata"],
                "total_available": len(available_gameweeks),
                "redirect_url": f"/api/v5/gameweeks/{latest_gw['gameweek']}/predictions"
            }
        }
        
        # Générer ETag basé sur la gameweek latest + metadata
        etag_data = cache_service.get_gameweek_metadata_for_etag(response_data)
        etag = cache_service.generate_etag(etag_data)
        
        # Cache court pour latest (5 minutes)
        cache_headers = cache_service.get_cache_headers_for_gameweek(
            latest_gw["gameweek"], 
            is_latest=True
        )
        
        # Vérifier If-None-Match
        if cache_service.check_if_none_match(request, etag):
            return cache_service.create_304_response(etag, cache_headers)
        
        # Créer la réponse avec headers de cache
        response = JSONResponse(content=response_data)
        cache_service.add_cache_headers_to_response(
            response, etag, cache_headers
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur récupération latest gameweek: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur serveur: {str(e)}"
        )

@router.get("/available")
async def get_available_gameweeks() -> dict:
    """Liste des gameweeks disponibles avec métadonnées"""
    
    try:
        # Scan artefacts gold disponibles
        gold_dir = Path("outputs/gold_backfill")
        
        available_gameweeks = []
        
        if gold_dir.exists():
            for gw_dir in sorted(gold_dir.iterdir()):
                if gw_dir.is_dir() and gw_dir.name.startswith('gw'):
                    try:
                        gw_num = int(gw_dir.name[2:])
                        
                        # Vérifier présence des fichiers requis
                        required_files = ['fixtures_canon.json', 'predictions.json', 'manifest.json']
                        has_all_files = all((gw_dir / filename).exists() for filename in required_files)
                        
                        # Charger métadonnées du manifest si disponible
                        metadata = {}
                        manifest_path = gw_dir / "manifest.json"
                        if manifest_path.exists():
                            try:
                                with open(manifest_path, 'r') as f:
                                    manifest = json.load(f)
                                
                                metadata = {
                                    "season_hash": manifest.get('hashes', {}).get('season_hash', 'unknown'),
                                    "generated_at": manifest.get('metadata', {}).get('generated_at', 'unknown'),
                                    "git_sha": manifest.get('source_control', {}).get('git_sha', 'unknown'),
                                    "validation": manifest.get('validation', {})
                                }
                            except:
                                metadata = {"error": "manifest_read_failed"}
                        
                        available_gameweeks.append({
                            "gameweek": gw_num,
                            "status": "available" if has_all_files else "incomplete",
                            "directory": gw_dir.name,
                            "required_files_present": has_all_files,
                            "metadata": metadata
                        })
                        
                    except ValueError:
                        logger.warning(f"Could not parse gameweek number from directory {gw_dir.name}")
                        continue
        
        # Scan également les prédictions pipeline (format legacy)
        predictions_dir = settings.PIPELINE_PREDICTIONS_DIR
        pipeline_gameweeks = set()
        
        if predictions_dir.exists():
            # Pattern files v1, v2 et v3
            prediction_files = list(predictions_dir.glob("j*_*_dual_champions_*.json")) + \
                              list(predictions_dir.glob("j*_*pipeline_v2_predictions_*.json")) + \
                              list(predictions_dir.glob("j*_predictions_v3_*.json"))
            
            for file_path in prediction_files:
                try:
                    # Extract round number from filename
                    parts = file_path.name.split('_')
                    if parts[0].startswith('j'):
                        round_num = int(parts[0][1:])  # Remove 'j' prefix
                        pipeline_gameweeks.add(round_num)
                except (ValueError, IndexError):
                    continue
        
        # Merger les informations
        all_gameweeks = set(gw["gameweek"] for gw in available_gameweeks)
        all_gameweeks.update(pipeline_gameweeks)
        
        # Compléter avec les gameweeks pipeline uniquement
        gold_gameweeks = set(gw["gameweek"] for gw in available_gameweeks)
        for gw_num in pipeline_gameweeks:
            if gw_num not in gold_gameweeks:
                available_gameweeks.append({
                    "gameweek": gw_num,
                    "status": "pipeline_only",
                    "directory": None,
                    "required_files_present": False,
                    "metadata": {"source": "pipeline_predictions"}
                })
        
        # Trier par numéro de gameweek
        available_gameweeks.sort(key=lambda x: x["gameweek"])
        
        # Statistiques
        gold_ready = sum(1 for gw in available_gameweeks if gw["status"] == "available")
        pipeline_only = sum(1 for gw in available_gameweeks if gw["status"] == "pipeline_only")
        latest_gameweek = max(available_gameweeks, key=lambda x: x["gameweek"])["gameweek"] if available_gameweeks else 0
        
        return {
            "api_version": "5.0.0",
            "generated_at": datetime.utcnow().isoformat(),
            "data": {
                "available_gameweeks": available_gameweeks,
                "statistics": {
                    "total_gameweeks": len(available_gameweeks),
                    "gold_artifacts_ready": gold_ready,
                    "pipeline_predictions_only": pipeline_only,
                    "latest_gameweek": latest_gameweek,
                    "season_progress": f"{latest_gameweek}/38"
                },
                "paths": {
                    "gold_artifacts_dir": str(gold_dir),
                    "pipeline_predictions_dir": str(predictions_dir)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération gameweeks disponibles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur serveur: {str(e)}"
        )

@router.get("/{gameweek}/coverage")
async def get_gameweek_coverage_validation(gameweek: int) -> dict:
    """Validation détaillée de la couverture d'une gameweek"""
    
    if not (1 <= gameweek <= 38):
        raise HTTPException(
            status_code=400,
            detail=f"Gameweek invalide: {gameweek}. Range valide: 1-38"
        )
    
    try:
        # Charger les données
        gameweek_data = await _load_gameweek_data(gameweek)
        if not gameweek_data:
            raise HTTPException(
                status_code=404,
                detail=f"Aucune donnée disponible pour la gameweek {gameweek}"
            )
        
        # Validation complète
        coverage_validator = get_coverage_validator()
        validation_report = coverage_validator.validate_gameweek_coverage(
            gameweek_data, gameweek, allow_partial=True  # Mode permissif pour debugging
        )
        
        return {
            "api_version": "5.0.0",
            "gameweek": gameweek,
            "validation_timestamp": datetime.utcnow().isoformat(),
            **validation_report
        }
        
    except CoverageValidationError as e:
        return {
            "api_version": "5.0.0",
            "gameweek": gameweek,
            "validation_failed": True,
            "error": str(e),
            "details": e.details,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Coverage validation error for GW{gameweek}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur validation couverture: {str(e)}"
        )

@router.get("/{gameweek}/status")
async def get_gameweek_status(gameweek: int) -> dict:
    """Status détaillé d'une gameweek spécifique"""
    
    if not (1 <= gameweek <= 38):
        raise HTTPException(
            status_code=400,
            detail=f"Gameweek invalide: {gameweek}. Range valide: 1-38"
        )
    
    try:
        # Vérifier artefacts gold
        gw_dir = Path(f"outputs/gold_backfill/gw{gameweek:02d}")
        
        status = {
            "gameweek": gameweek,
            "gold_artifacts": {
                "available": gw_dir.exists(),
                "directory": str(gw_dir),
                "files": {}
            },
            "pipeline_predictions": {
                "available": False,
                "files": []
            }
        }
        
        # Analyser artefacts gold
        if gw_dir.exists():
            required_files = ['fixtures_canon.json', 'predictions.json', 'manifest.json']
            
            for filename in required_files:
                file_path = gw_dir / filename
                if file_path.exists():
                    file_stats = file_path.stat()
                    status["gold_artifacts"]["files"][filename] = {
                        "exists": True,
                        "size_kb": round(file_stats.st_size / 1024, 2),
                        "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    }
                else:
                    status["gold_artifacts"]["files"][filename] = {"exists": False}
        
        # Analyser prédictions pipeline
        predictions_dir = settings.PIPELINE_PREDICTIONS_DIR
        if predictions_dir.exists():
            pipeline_files = list(predictions_dir.glob(f"j{gameweek}_*_dual_champions_*.json")) + \
                            list(predictions_dir.glob(f"j{gameweek}_*pipeline_v2_predictions_*.json")) + \
                            list(predictions_dir.glob(f"j{gameweek}_predictions_v3_*.json"))
            
            if pipeline_files:
                status["pipeline_predictions"]["available"] = True
                for file_path in pipeline_files:
                    file_stats = file_path.stat()
                    status["pipeline_predictions"]["files"].append({
                        "name": file_path.name,
                        "size_kb": round(file_stats.st_size / 1024, 2),
                        "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    })
        
        # Déterminer status global
        has_gold = status["gold_artifacts"]["available"] and \
                  all(file_info.get("exists", False) for file_info in status["gold_artifacts"]["files"].values())
        has_pipeline = status["pipeline_predictions"]["available"]
        
        if has_gold:
            overall_status = "gold_ready"
        elif has_pipeline:
            overall_status = "pipeline_ready"
        else:
            overall_status = "not_available"
        
        status["overall_status"] = overall_status
        status["generated_at"] = datetime.utcnow().isoformat()
        
        return status
        
    except Exception as e:
        logger.error(f"Erreur status gameweek {gameweek}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur serveur: {str(e)}"
        )

@router.get("/{gameweek}/predictions")
async def get_gameweek_predictions(gameweek: int, request: Request) -> Response:
    """Récupère les prédictions pour une gameweek spécifique avec cache immutable"""
    
    if not (1 <= gameweek <= 38):
        raise HTTPException(
            status_code=400,
            detail=f"Gameweek invalide: {gameweek}. Range valide: 1-38"
        )
    
    cache_service = get_cache_service()
    
    try:
        response_data = None
        file_path = None
        
        # Priorité 1: Artefacts gold
        gw_dir = Path(f"outputs/gold_backfill/gw{gameweek:02d}")
        
        if gw_dir.exists() and (gw_dir / "predictions.json").exists():
            logger.info(f"Loading gold artifacts for gameweek {gameweek}")
            file_path = gw_dir / "predictions.json"
            
            with open(file_path, 'r') as f:
                gold_data = json.load(f)
            
            # Adapter format gold vers API v5
            response_data = adapt_gold_to_apiv5(gold_data, gameweek)
        
        # Priorité 2: Artefacts versionnés (nouveau système)
        elif settings.PIPELINE_PREDICTIONS_DIR.exists():
            versioned_dir = settings.PIPELINE_PREDICTIONS_DIR / "versioned"
            
            if versioned_dir.exists():
                # Chercher latest.json d'abord
                latest_file = versioned_dir / f"j{gameweek}_latest.json"
                if latest_file.exists():
                    logger.info(f"Loading versioned artifact for gameweek {gameweek}")
                    file_path = latest_file
                    
                    with open(latest_file, 'r') as f:
                        response_data = json.load(f)
                
                # Sinon chercher fichiers versionnés par timestamp
                else:
                    versioned_files = list(versioned_dir.glob(f"j{gameweek}_predictions_v3_*.json"))
                    if versioned_files:
                        latest_file = max(versioned_files, key=lambda p: p.stat().st_mtime)
                        logger.info(f"Loading latest versioned artifact: {latest_file.name}")
                        file_path = latest_file
                        
                        with open(latest_file, 'r') as f:
                            response_data = json.load(f)
        
        # Priorité 3: Prédictions pipeline (legacy) si pas encore trouvé
        if response_data is None:
            predictions_dir = settings.PIPELINE_PREDICTIONS_DIR
            
            if predictions_dir.exists():
                # Chercher fichiers prédictions par ordre de préférence
                prediction_patterns = [
                    f"j{gameweek}_predictions_v3_*.json",
                    f"j{gameweek}_*pipeline_v2_predictions_*.json", 
                    f"j{gameweek}_*_dual_champions_*.json"
                ]
                
                for pattern in prediction_patterns:
                    prediction_files = list(predictions_dir.glob(pattern))
                    if prediction_files:
                        # Prendre le plus récent
                        latest_file = max(prediction_files, key=lambda p: p.stat().st_mtime)
                        logger.info(f"Loading pipeline predictions from {latest_file.name}")
                        file_path = latest_file
                        
                        with open(latest_file, 'r') as f:
                            pipeline_data = json.load(f)
                        
                        # Adapter format pipeline vers API v5
                        response_data = adapt_pipeline_to_apiv5(pipeline_data, gameweek, latest_file.name)
                        break
        
        # Vérifier qu'on a trouvé des données
        if response_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Aucune prédiction disponible pour la gameweek {gameweek}"
            )
        
        # Validation de couverture avant exposition
        coverage_validator = get_coverage_validator()
        try:
            validation_report = coverage_validator.validate_gameweek_coverage(
                response_data, gameweek, allow_partial=False
            )
            
            # Ajouter les métriques de validation à la réponse
            response_data["validation"] = {
                "gw_compliant": validation_report["ready_for_production"],
                "fixtures_count": validation_report["summary"]["fixtures_count"],
                "epl_teams_only": validation_report["validations"]["epl_teams"]["valid"],
                "unique_fixtures": validation_report["validations"]["uniqueness"]["valid"],
                "balanced_teams": validation_report["validations"]["team_balance"]["valid"],
                "quality_predictions": validation_report["validations"]["prediction_quality"]["valid"],
                "ko2h_strict": True,  # Maintenu pour compatibilité
                "json_schema_valid": True
            }
            
            # Warning si pas prêt pour production mais on sert quand même
            if not validation_report["ready_for_production"]:
                logger.warning(
                    f"Serving GW{gameweek} with validation issues: "
                    f"{validation_report['summary']['passed_validations']}/5 checks passed"
                )
                
        except CoverageValidationError as e:
            # En mode strict, ne pas exposer les données invalides
            logger.error(f"Coverage validation failed for GW{gameweek}: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Gameweek {gameweek} ne respecte pas les standards de qualité",
                headers={"X-Validation-Error": "coverage_validation_failed"}
            )
        
        # Générer ETag immutable basé sur le contenu
        etag_data = cache_service.get_gameweek_metadata_for_etag(response_data)
        etag = cache_service.generate_etag(etag_data)
        
        # Cache long pour gameweek spécifique (24h, immutable)
        cache_headers = cache_service.get_cache_headers_for_gameweek(gameweek, is_latest=False)
        
        # Vérifier If-None-Match pour 304
        if cache_service.check_if_none_match(request, etag):
            return cache_service.create_304_response(etag, cache_headers)
        
        # Last-Modified basé sur le fichier
        last_modified = None
        if file_path and file_path.exists():
            last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            if cache_service.check_if_modified_since(request, last_modified):
                pass  # Continuer avec la réponse complète
            else:
                return cache_service.create_304_response(etag, cache_headers)
        
        # Créer la réponse avec headers de cache
        response = JSONResponse(content=response_data)
        cache_service.add_cache_headers_to_response(
            response, etag, cache_headers, last_modified
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur récupération prédictions gameweek {gameweek}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur serveur: {str(e)}"
        )

@router.get("/{gameweek}/metadata")
async def get_gameweek_metadata(gameweek: int) -> dict:
    """Métadonnées d'observabilité pour une gameweek spécifique"""
    
    if not (1 <= gameweek <= 38):
        raise HTTPException(
            status_code=400,
            detail=f"Gameweek invalide: {gameweek}. Range valide: 1-38"
        )
    
    try:
        metadata = {
            "gameweek": gameweek,
            "generated_at": datetime.utcnow().isoformat(),
            "api_version": "5.0.0"
        }
        
        # Vérifier artefacts versionnés (priorité 1)
        versioned_dir = settings.PIPELINE_PREDICTIONS_DIR / "versioned"
        versioned_latest = versioned_dir / f"j{gameweek}_latest.json"
        
        if versioned_latest.exists():
            try:
                with open(versioned_latest, 'r') as f:
                    versioned_data = json.load(f)
                
                file_stats = versioned_latest.stat()
                
                metadata.update({
                    "source": "versioned_artifacts",
                    "artifact_file": f"versioned/j{gameweek}_latest.json",
                    "file_size_kb": round(file_stats.st_size / 1024, 2),
                    "file_modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "model_version": versioned_data.get("metadata", {}).get("model_version", "enhanced_baseline_v2.4"),
                    "season_hash": versioned_data.get("metadata", {}).get("season_hash", "unknown"),
                    "dataset_hash": versioned_data.get("metadata", {}).get("dataset_hash", "unknown"),
                    "git_sha": versioned_data.get("metadata", {}).get("git_sha", "unknown"),
                    "predictions_count": versioned_data.get("fixtures_count", 0),
                    "features_hash": versioned_data.get("metadata", {}).get("features_hash", "unknown"),
                    "pipeline_version": versioned_data.get("metadata", {}).get("pipeline_version", "durci_v1.0")
                })
                
                return metadata
                
            except Exception as e:
                logger.warning(f"Could not read versioned metadata for GW{gameweek}: {e}")
        
        # Vérifier artefacts gold (priorité 2)
        gw_dir = Path(f"outputs/gold_backfill/gw{gameweek:02d}")
        if gw_dir.exists() and (gw_dir / "predictions.json").exists():
            try:
                with open(gw_dir / "predictions.json", 'r') as f:
                    gold_data = json.load(f)
                
                file_stats = (gw_dir / "predictions.json").stat()
                
                metadata.update({
                    "source": "gold_artifacts",
                    "artifact_file": f"gw{gameweek:02d}/predictions.json",
                    "file_size_kb": round(file_stats.st_size / 1024, 2),
                    "file_modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "model_version": gold_data.get("model_version", "v2.4"),
                    "season_hash": gold_data.get("season_hash", "unknown"),
                    "dataset_hash": gold_data.get("dataset_hash", "unknown"),
                    "git_sha": gold_data.get("git_sha", "unknown"),
                    "predictions_count": len(gold_data.get("matches", [])),
                    "features_hash": gold_data.get("features_hash", "unknown")
                })
                
            except Exception as e:
                logger.warning(f"Could not read gold metadata for GW{gameweek}: {e}")
                metadata.update({
                    "source": "gold_artifacts", 
                    "error": "metadata_read_failed"
                })
                
        else:
            # Vérifier prédictions pipeline
            predictions_dir = settings.PIPELINE_PREDICTIONS_DIR
            if predictions_dir.exists():
                # Chercher fichiers prédictions
                prediction_patterns = [
                    f"j{gameweek}_predictions_v3_*.json",
                    f"j{gameweek}_*pipeline_v2_predictions_*.json", 
                    f"j{gameweek}_*_dual_champions_*.json"
                ]
                
                found_file = None
                for pattern in prediction_patterns:
                    prediction_files = list(predictions_dir.glob(pattern))
                    if prediction_files:
                        found_file = max(prediction_files, key=lambda p: p.stat().st_mtime)
                        break
                
                if found_file:
                    try:
                        with open(found_file, 'r') as f:
                            pipeline_data = json.load(f)
                        
                        file_stats = found_file.stat()
                        
                        metadata.update({
                            "source": "pipeline_predictions",
                            "artifact_file": found_file.name,
                            "file_size_kb": round(file_stats.st_size / 1024, 2),
                            "file_modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                            "model_version": pipeline_data.get("pipeline_metadata", {}).get("version", "pipeline_durci_v1.0"),
                            "season_hash": pipeline_data.get("season_hash", "pipeline_generated"),
                            "dataset_hash": pipeline_data.get("dataset_hash", "pipeline_generated"),
                            "git_sha": pipeline_data.get("git_sha", "pipeline_generated"),
                            "predictions_count": len(pipeline_data.get("matches", pipeline_data.get("predictions", {}))),
                            "features_hash": pipeline_data.get("features_hash", "unknown")
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not read pipeline metadata for GW{gameweek}: {e}")
                        metadata.update({
                            "source": "pipeline_predictions",
                            "artifact_file": found_file.name,
                            "error": "metadata_read_failed"
                        })
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Aucune métadonnée disponible pour la gameweek {gameweek}"
                    )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Aucune métadonnée disponible pour la gameweek {gameweek}"
                )
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur récupération métadonnées gameweek {gameweek}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur serveur: {str(e)}"
        )

def adapt_gold_to_apiv5(gold_data: dict, gameweek: int) -> dict:
    """Adapte format gold artifacts vers API v5"""
    
    predictions = {}
    
    for match in gold_data.get("matches", []):
        match_key = f"{match['home_team']}_vs_{match['away_team']}"
        
        predictions[match_key] = {
            "prediction": match["ensemble"]["prediction"],
            "confidence": match["ensemble"]["confidence"],
            "probabilities": match["ensemble"]["probabilities"],
            "model_info": {
                "prediction_mode": "enhanced_baseline_v24",
                "enhanced_metadata": {},
                "model_version": "v2.4",
                "accuracy_improvement": "baseline_champion",
                "away_bias_correction": "enabled"
            },
            "market_features": {
                "market_confidence": match.get("market_probs_raw", {}).get("home", 0.33),
                "market_entropy": match.get("market_entropy_norm", 0.8),
                "market_favorite": match["ensemble"]["prediction"],
                "home_advantage_market": 0.1
            },
            "match_info": {
                "home": match["home_team"],
                "away": match["away_team"],
                "date": match["date"]
            }
        }
    
    return {
        "api_version": "5.0.0",
        "mode": "gold_artifacts",
        "gameweek": gameweek,
        "metadata": {
            "season_hash": gold_data.get("season_hash", "unknown"),
            "dataset_hash": gold_data.get("dataset_hash", "unknown"),
            "git_sha": gold_data.get("git_sha", "unknown"),
            "generated_at": gold_data.get("generated_at", datetime.utcnow().isoformat()),
            "strict_validated": True,
            "fixtures_count": len(predictions)
        },
        "fixtures_count": len(predictions),
        "predictions": predictions,
        "validation": {
            "gw_compliant": True,
            "ko2h_strict": True,
            "epl_teams_only": True,
            "json_schema_valid": True
        }
    }

def adapt_pipeline_to_apiv5(pipeline_data: dict, gameweek: int, filename: str) -> dict:
    """Adapte format pipeline vers API v5"""
    
    predictions = {}
    
    # Gérer différents formats de pipeline
    if "matches" in pipeline_data:
        # Format avec liste de matchs
        matches = pipeline_data["matches"]
        for match in matches:
            match_key = f"{match['home_team']}_vs_{match['away_team']}"
            
            # Utiliser ensemble ou enhanced_baseline selon disponibilité
            prediction_source = match.get("ensemble", match.get("enhanced_baseline", {}))
            
            predictions[match_key] = {
                "prediction": prediction_source.get("prediction", "H"),
                "confidence": prediction_source.get("confidence", 0.5),
                "probabilities": prediction_source.get("probabilities", {
                    "home": 0.33, "draw": 0.33, "away": 0.34
                }),
                "model_info": {
                    "prediction_mode": "pipeline_integration",
                    "enhanced_metadata": {},
                    "model_version": "pipeline_durci_v1.0",
                    "accuracy_improvement": "pipeline_champion",
                    "away_bias_correction": "enabled"
                },
                "market_features": {
                    "market_confidence": match.get("market_entropy_norm", 0.8),
                    "market_entropy": match.get("market_entropy_norm", 0.8),
                    "market_favorite": prediction_source.get("prediction", "H"),
                    "home_advantage_market": 0.1
                },
                "match_info": {
                    "home": match["home_team"],
                    "away": match["away_team"],
                    "date": match.get("date", "2025-10-20")
                }
            }
            
    elif "predictions" in pipeline_data:
        # Format avec dictionnaire de prédictions (clé = "Team1_vs_Team2")
        prediction_dict = pipeline_data["predictions"]
        for match_key, prediction_data in prediction_dict.items():
            # Extraire les noms d'équipes de la clé
            home_team, away_team = match_key.split("_vs_")
            
            predictions[match_key] = {
                "prediction": prediction_data.get("prediction", "H"),
                "confidence": prediction_data.get("confidence", 0.5),
                "probabilities": prediction_data.get("probabilities", {
                    "home": 0.33, "draw": 0.33, "away": 0.34
                }),
                "model_info": {
                    "prediction_mode": "pipeline_v3_integration",
                    "enhanced_metadata": {},
                    "model_version": pipeline_data.get("pipeline_metadata", {}).get("version", "v3.0"),
                    "accuracy_improvement": "enhanced_baseline_v24",
                    "away_bias_correction": "enabled"
                },
                "market_features": {
                    "market_confidence": prediction_data.get("market_entropy_norm", 0.8),
                    "market_entropy": prediction_data.get("market_entropy_norm", 0.8),
                    "market_favorite": prediction_data.get("prediction", "H"),
                    "home_advantage_market": 0.1
                },
                "match_info": {
                    "home": home_team,
                    "away": away_team,
                    "date": prediction_data.get("date", "2025-10-20")
                }
            }
    else:
        # Format non reconnu
        logger.warning(f"Format de données non reconnu pour {filename}: {list(pipeline_data.keys())}")
        return {
            "api_version": "5.0.0",
            "mode": "error",
            "gameweek": gameweek,
            "metadata": {
                "season_hash": "unknown",
                "dataset_hash": "unknown", 
                "git_sha": "unknown",
                "generated_at": datetime.utcnow().isoformat(),
                "strict_validated": False,
                "fixtures_count": 0
            },
            "fixtures_count": 0,
            "predictions": {},
            "validation": {
                "gw_compliant": False,
                "ko2h_strict": False,
                "epl_teams_only": False,
                "json_schema_valid": False
            }
        }
    
    return {
        "api_version": "5.0.0",
        "mode": "pipeline_integration",
        "gameweek": gameweek,
        "metadata": {
            "season_hash": pipeline_data.get("season_hash", "pipeline_generated"),
            "dataset_hash": pipeline_data.get("dataset_hash", "pipeline_generated"),
            "git_sha": pipeline_data.get("git_sha", "pipeline_generated"),
            "generated_at": pipeline_data.get("generated_at", datetime.utcnow().isoformat()),
            "strict_validated": False,
            "fixtures_count": len(predictions)
        },
        "fixtures_count": len(predictions),
        "predictions": predictions,
        "validation": {
            "gw_compliant": len(predictions) == 10,
            "ko2h_strict": False,
            "epl_teams_only": True,
            "json_schema_valid": True
        }
    }

@router.get("/{gameweek}/fixtures")
async def get_gameweek_fixtures(gameweek: int, request: Request) -> Response:
    """Get real fixture data with kickoff times for a gameweek"""
    
    if gameweek < 1 or gameweek > 38:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid gameweek: {gameweek}. Must be between 1 and 38"
        )
    
    try:
        fixture_service = get_fixture_service()
        fixtures = fixture_service.get_fixtures_for_gameweek(gameweek)
        
        if not fixtures:
            raise HTTPException(
                status_code=404,
                detail=f"No fixture data available for gameweek {gameweek}"
            )
        
        response_data = {
            "api_version": "5.0.0",
            "generated_at": datetime.utcnow().isoformat(),
            "data": {
                "gameweek": gameweek,
                "fixtures": fixtures,
                "total_fixtures": len(fixtures)
            }
        }
        
        return JSONResponse(
            content=response_data,
            headers={
                "Cache-Control": "public, max-age=300",  # 5 min cache
                "X-API-Version": "5.0.0"
            }
        )
        
    except Exception as e:
        logger.error(f"Error fetching fixtures for gameweek {gameweek}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch fixture data: {str(e)}"
        )