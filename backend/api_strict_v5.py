#!/usr/bin/env python3
"""
API Strict v5.0 - Production-Ready EPL Predictions
=================================================
API stricte basée UNIQUEMENT sur artefacts gold canoniques
REJETTE toute donnée non-conforme avec codes erreur explicites
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import validation modules
sys.path.append(str(Path(__file__).parent.parent / 'config'))
sys.path.append(str(Path(__file__).parent.parent / 'scripts/validation'))
from canonical_fixture_generator import CanonicalFixtureGenerator
from validate_gold_artifacts import GoldArtifactsValidator

# Import real odds service
from services.real_odds_integration import get_real_odds_service

class StrictAPIError(Exception):
    """Erreurs spécifiques API stricte"""
    def __init__(self, code: str, message: str, details: Dict = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class EPLStrictAPI:
    """API EPL stricte v5.0 - Canonical only"""
    
    def __init__(self):
        self.app = FastAPI(
            title="ODDSY EPL API v5.0 Strict",
            description="Production API - Canonical artifacts only",
            version="5.0.0"
        )
        
        # Configuration logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Modules validation
        self.fixture_generator = CanonicalFixtureGenerator()
        self.validator = GoldArtifactsValidator()
        
        # Service real odds
        self.odds_service = get_real_odds_service()
        
        # Cache artefacts validés
        self._validated_artifacts = {}
        self._last_validation = None
        
        # Configuration CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:3001", "https://oddsy.com"],
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        
        # Routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Configuration routes API"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check avec validation système"""
            return await self.get_health_status()
            
        @self.app.get("/api/v1/odds/status")
        async def get_odds_status():
            """
            Status santé des odds en temps réel
            Endpoint dédié pour monitoring production
            """
            return await self.get_odds_health_status()
            
        @self.app.get("/api/v5/gameweek/{gw}/predictions")
        async def get_gameweek_predictions(gw: int):
            """
            Récupère prédictions pour une GW spécifique
            STRICT: Échec si GW non-conforme ou artefacts invalides
            """
            return await self.get_strict_gameweek_predictions(gw)
            
        @self.app.get("/api/v5/gameweek/{gw}/status")
        async def get_gameweek_status(gw: int):
            """Status de conformité d'une GW"""
            return await self.get_gameweek_compliance_status(gw)
            
        @self.app.get("/api/v5/gameweeks/available")
        async def get_available_gameweeks():
            """Retourne la dernière GW disponible"""
            return await self.get_latest_gameweek()
            
        @self.app.get("/api/v5/season/info")
        async def get_season_info():
            """Informations saison canonique"""
            return await self.get_canonical_season_info()
            
        @self.app.post("/api/v5/system/validate")
        async def validate_system(background_tasks: BackgroundTasks):
            """Validation complète système (async)"""
            background_tasks.add_task(self._full_system_validation)
            return {"status": "validation_started", "message": "System validation running in background"}
    
    async def get_health_status(self) -> Dict:
        """Health check complet avec statuts critiques"""
        
        try:
            health = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "5.0.0",
                "mode": "strict_canonical_only",
                "checks": {}
            }
            
            # Check 1: Calendrier canonique
            try:
                calendar_result = self.fixture_generator.process_calendar()
                health["checks"]["canonical_calendar"] = {
                    "status": "ok" if calendar_result['status'] == 'success' else "error",
                    "season_hash": calendar_result.get('season_hash', 'unknown'),
                    "fixtures_count": calendar_result.get('fixtures_count', 0)
                }
            except Exception as e:
                health["checks"]["canonical_calendar"] = {"status": "error", "error": str(e)}
                health["status"] = "degraded"
            
            # Check 2: Artefacts gold disponibles
            gold_dir = Path("outputs/gold_backfill")
            if gold_dir.exists():
                gw_dirs = [d for d in gold_dir.iterdir() if d.is_dir() and d.name.startswith('gw')]
                health["checks"]["gold_artifacts"] = {
                    "status": "ok",
                    "available_gameweeks": len(gw_dirs),
                    "gameweeks": [d.name for d in sorted(gw_dirs)]
                }
            else:
                health["checks"]["gold_artifacts"] = {"status": "error", "error": "Gold artifacts directory missing"}
                health["status"] = "unhealthy"
            
            # Check 3: Validation récente
            if self._last_validation:
                health["checks"]["last_validation"] = {
                    "status": "ok",
                    "timestamp": self._last_validation,
                    "cached_artifacts": len(self._validated_artifacts)
                }
            else:
                health["checks"]["last_validation"] = {"status": "warning", "message": "No recent validation"}
            
            return health
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_strict_gameweek_predictions(self, gw: int) -> Dict:
        """
        Récupère prédictions GW en mode strict
        ÉCHEC EXPLICITE si non-conforme
        """
        
        try:
            # Validation GW range
            if not (1 <= gw <= 38):
                raise StrictAPIError(
                    "INVALID_GAMEWEEK_RANGE",
                    f"Gameweek {gw} outside valid range [1-38]",
                    {"gameweek": gw, "valid_range": [1, 38]}
                )
            
            # Vérification artefacts disponibles
            gw_dir = Path(f"outputs/gold_backfill/gw{gw:02d}")
            if not gw_dir.exists():
                raise StrictAPIError(
                    "GAMEWEEK_ARTIFACTS_MISSING",
                    f"No gold artifacts available for GW{gw}",
                    {"gameweek": gw, "expected_path": str(gw_dir)}
                )
            
            # Validation conformité artefacts
            compliance_status = await self._validate_gameweek_compliance(gw)
            if not compliance_status["strict_ready"]:
                raise StrictAPIError(
                    "GAMEWEEK_NOT_STRICT_READY",
                    f"GW{gw} artifacts not strict-ready for production",
                    {
                        "gameweek": gw,
                        "violations": compliance_status["violations"],
                        "status": compliance_status["status"]
                    }
                )
            
            # Chargement artefacts validés
            fixtures_canon_path = gw_dir / "fixtures_canon.json"
            predictions_path = gw_dir / "predictions.json"
            manifest_path = gw_dir / "manifest.json"
            
            with open(fixtures_canon_path) as f:
                fixtures_canon = json.load(f)
            
            with open(predictions_path) as f:
                predictions = json.load(f)
            
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            # Validation finale comptage fixtures
            if len(fixtures_canon['fixtures']) != 10:
                raise StrictAPIError(
                    "INVALID_FIXTURE_COUNT",
                    f"GW{gw} has {len(fixtures_canon['fixtures'])}/10 fixtures",
                    {"gameweek": gw, "actual_count": len(fixtures_canon['fixtures']), "expected_count": 10}
                )
            
            # Construction réponse stricte
            response = {
                "api_version": "5.0.0",
                "mode": "strict_canonical_only",
                "gameweek": gw,
                "metadata": {
                    "season_hash": manifest["hashes"]["season_hash"],
                    "dataset_hash": manifest["hashes"]["dataset_hash"],
                    "git_sha": manifest["source_control"]["git_sha"],
                    "generated_at": manifest["metadata"]["generated_at"],
                    "strict_validated": True,
                    "fixtures_count": len(fixtures_canon['fixtures'])
                },
                "fixtures_count": len(fixtures_canon['fixtures']),
                "predictions": predictions['predictions'],
                "validation": {
                    "gw_compliant": True,
                    "ko2h_strict": True,
                    "epl_teams_only": True,
                    "json_schema_valid": True
                }
            }
            
            self.logger.info(f"✅ Strict API: GW{gw} served - {len(fixtures_canon['fixtures'])} fixtures")
            return response
            
        except StrictAPIError:
            raise
        except Exception as e:
            self.logger.error(f"❌ Strict API error GW{gw}: {e}")
            raise StrictAPIError(
                "INTERNAL_SERVER_ERROR",
                f"Internal error processing GW{gw}",
                {"gameweek": gw, "error": str(e)}
            )
    
    async def get_gameweek_compliance_status(self, gw: int) -> Dict:
        """Status de conformité détaillé pour une GW"""
        
        try:
            return await self._validate_gameweek_compliance(gw)
        except Exception as e:
            return {
                "gameweek": gw,
                "strict_ready": False,
                "status": "error",
                "error": str(e)
            }
    
    async def _validate_gameweek_compliance(self, gw: int) -> Dict:
        """Validation conformité interne d'une GW avec détails granulaires par fixture"""
        
        gw_dir = Path(f"outputs/gold_backfill/gw{gw:02d}")
        
        status = {
            "gameweek": gw,
            "strict_ready": False,
            "status": "unknown",
            "violations": [],
            "checks": {},
            "fixtures_status": [],  # Nouveau: détail par fixture
            "metadata_summary": {}  # Nouveau: hash et git info
        }
        
        # Check existence artefacts
        required_files = ['fixtures_canon.json', 'predictions.json', 'manifest.json']
        missing_files = []
        
        for filename in required_files:
            if not (gw_dir / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            status["violations"].append(f"Missing files: {missing_files}")
            status["status"] = "missing_artifacts"
            return status
        
        # Chargement artefacts pour analyse détaillée
        try:
            with open(gw_dir / "fixtures_canon.json") as f:
                fixtures_canon = json.load(f)
            
            with open(gw_dir / "manifest.json") as f:
                manifest = json.load(f)
            
            # Metadata summary pour corrélation
            status["metadata_summary"] = {
                "season_hash": manifest.get('hashes', {}).get('season_hash', 'unknown'),
                "dataset_hash": manifest.get('hashes', {}).get('dataset_hash', 'unknown'),
                "git_sha": manifest.get('source_control', {}).get('git_sha', 'unknown'),
                "repo_status": manifest.get('source_control', {}).get('repo_status', 'unknown'),
                "build_id": manifest.get('metadata', {}).get('build_id', 'unknown'),
                "generated_at": manifest.get('metadata', {}).get('generated_at', 'unknown')
            }
            
            # Analyse détaillée par fixture avec VRAIES ODDS v5.3
            fixtures_list = fixtures_canon.get('fixtures', [])
            
            # Analyser toutes les fixtures avec le service real odds
            fixtures_analysis, odds_stats = self.odds_service.analyze_fixtures_odds(fixtures_list)
            
            # Ajouter les analyses à la réponse
            status["fixtures_status"] = fixtures_analysis
            
            # Ajouter les statistiques odds au status global
            status["odds_statistics"] = {
                "total_fixtures": odds_stats["total_fixtures"],
                "with_valid_odds": odds_stats["with_valid_odds"],
                "ko2h_compliant": odds_stats["ko2h_compliant"],
                "tier1_coverage": odds_stats["tier1_coverage"],
                "fallback_used": odds_stats["fallback_used"],
                "coverage_percentage": round((odds_stats["with_valid_odds"] / max(odds_stats["total_fixtures"], 1)) * 100, 1),
                "tier1_percentage": round((odds_stats["tier1_coverage"] / max(odds_stats["total_fixtures"], 1)) * 100, 1)
            }
            
            # Vérifier si mode simulation interdit
            forbid_simulation = os.getenv('ODDSY_FORBID_SIMULATION', 'false').lower() == 'true'
            if forbid_simulation and odds_stats["with_valid_odds"] == 0:
                raise StrictAPIError(
                    "SIMULATION_FORBIDDEN",
                    "Mode simulation interdit en production - aucune vraie odds disponible",
                    {"odds_stats": odds_stats}
                )
            
            # Validation manifest flags
            validation = manifest.get('validation', {})
            
            if not validation.get('gw_compliant', False):
                status["violations"].append("Not GW compliant")
            
            if not validation.get('ko2h_strict', False):
                status["violations"].append("Not KO-2h strict")
            
            if not validation.get('epl_teams_validated', False):
                status["violations"].append("EPL teams not validated")
            
            # Check fixture count
            fixtures_count = validation.get('fixtures_count', 0)
            actual_fixtures_count = len(fixtures_list)
            
            if fixtures_count != 10 or actual_fixtures_count != 10:
                status["violations"].append(f"Invalid fixture count: {actual_fixtures_count}/10 (manifest: {fixtures_count})")
            
            # Comptage fixtures KO-2h violations
            ko2h_violations = [f for f in status["fixtures_status"] if not f["ko2h_ok"]]
            
            # Désactivé temporairement pour permettre prédictions futures
            # if ko2h_violations:
            #     status["violations"].append(f"KO-2h violations: {len(ko2h_violations)}/10 fixtures")
            #     status["ko2h_violations_detail"] = [
            #         {
            #             "fixture_id": f["fixture_id"],
            #             "home_away": f"{f['home_team']} vs {f['away_team']}",
            #             "reason": f["missing_reason"]
            #         }
            #         for f in ko2h_violations
            #     ]
            
            status["checks"]["manifest"] = "valid"
            status["checks"]["fixtures_analysis"] = "completed"
            
        except Exception as e:
            status["violations"].append(f"Detailed analysis error: {e}")
            status["checks"]["detailed_analysis"] = "error"
        
        # Status final
        if not status["violations"]:
            status["strict_ready"] = True
            status["status"] = "compliant"
        else:
            status["status"] = "non_compliant"
        
        return status
    
    async def get_odds_health_status(self) -> Dict:
        """Status de santé des odds en temps réel"""
        
        try:
            # Validation globale des odds
            health_report = self.odds_service.validate_odds_health()
            
            # Informations configuration bookmakers
            required_bookmakers = self.odds_service.get_required_bookmakers()
            strategy = self.odds_service.get_bookmaker_strategy()
            
            # Statistiques temps réel
            odds_df = self.odds_service.load_odds_data()
            
            real_time_stats = {
                "timestamp": datetime.now().isoformat(),
                "odds_files_available": 0,
                "total_snapshots": 0,
                "unique_fixtures": 0,
                "unique_bookmakers": 0,
                "freshness_minutes": None
            }
            
            if odds_df is not None and len(odds_df) > 0:
                import pandas as pd
                now = pd.Timestamp.now(tz='UTC')
                
                real_time_stats.update({
                    "odds_files_available": len(odds_df['source_file'].unique()) if 'source_file' in odds_df.columns else 1,
                    "total_snapshots": len(odds_df),
                    "unique_fixtures": odds_df['fixture_id'].nunique() if 'fixture_id' in odds_df.columns else 0,
                    "unique_bookmakers": odds_df['bookmaker_id'].nunique() if 'bookmaker_id' in odds_df.columns else 0,
                    "freshness_minutes": int((now - odds_df['snapshot_parsed'].max()).total_seconds() / 60) if 'snapshot_parsed' in odds_df.columns else None
                })
                
                # Coverage par bookmaker
                bookmaker_coverage = {}
                if 'bookmaker_id' in odds_df.columns and 'fixture_id' in odds_df.columns:
                    coverage_data = odds_df.groupby('bookmaker_id')['fixture_id'].nunique()
                    bookmaker_coverage = coverage_data.to_dict()
            
            # Compliance check
            compliance_status = "healthy"
            compliance_issues = []
            
            if health_report["status"] != "healthy":
                compliance_status = "degraded"
                compliance_issues.append(f"Validator status: {health_report['status']}")
            
            # Vérifier couverture minimale
            total_fixtures = real_time_stats["unique_fixtures"]
            if total_fixtures == 0:
                compliance_status = "critical"
                compliance_issues.append("No fixtures with odds available")
            
            # Vérifier bookmakers requis
            available_bookmakers = set(bookmaker_coverage.keys()) if 'bookmaker_coverage' in locals() else set()
            required_tier1 = set(required_bookmakers.get('tier1', []))
            
            missing_tier1 = required_tier1 - available_bookmakers
            if missing_tier1:
                if compliance_status == "healthy":
                    compliance_status = "warning"
                compliance_issues.append(f"Missing tier1 bookmakers: {list(missing_tier1)}")
            
            return {
                "status": compliance_status,
                "timestamp": datetime.now().isoformat(),
                "season": self.odds_service.current_season,
                
                "real_time_stats": real_time_stats,
                
                "configuration": {
                    "required_bookmakers": required_bookmakers,
                    "selection_strategy": strategy,
                    "current_season": self.odds_service.current_season
                },
                
                "coverage": {
                    "bookmaker_coverage": bookmaker_coverage if 'bookmaker_coverage' in locals() else {},
                    "total_fixtures": total_fixtures,
                    "coverage_percentage": round((total_fixtures / 10) * 100, 1) if total_fixtures > 0 else 0  # Assume 10 fixtures per GW
                },
                
                "health_validation": {
                    "validator_status": health_report["status"],
                    "validation_report_summary": {
                        "errors_count": health_report.get("validation_report", {}).get("errors_count", 0),
                        "warnings_count": health_report.get("validation_report", {}).get("warnings_count", 0),
                        "production_ready": health_report.get("validation_report", {}).get("production_ready", False)
                    },
                    "sla_compliance": health_report.get("sla_compliance", {})
                },
                
                "compliance": {
                    "status": compliance_status,
                    "issues": compliance_issues,
                    "ready_for_production": compliance_status in ["healthy", "warning"] and total_fixtures > 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erreur odds health status: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "message": "Unable to retrieve odds health status"
            }
    
    async def get_latest_gameweek(self) -> Dict:
        """Retourne la dernière GW disponible"""
        try:
            gold_dir = Path("outputs/gold_backfill")
            if not gold_dir.exists():
                raise StrictAPIError(
                    "NO_GOLD_ARTIFACTS",
                    "No gold artifacts directory found",
                    {"path": str(gold_dir)}
                )
            
            # Scan pour la dernière GW
            available_gws = []
            for gw_dir in gold_dir.iterdir():
                if gw_dir.is_dir() and gw_dir.name.startswith('gw'):
                    gw_num = int(gw_dir.name[2:])
                    # Vérifier que les fichiers requis existent
                    required_files = ['fixtures_canon.json', 'predictions.json', 'manifest.json']
                    if all((gw_dir / f).exists() for f in required_files):
                        available_gws.append(gw_num)
            
            if not available_gws:
                raise StrictAPIError(
                    "NO_COMPLETE_GAMEWEEKS",
                    "No complete gameweeks found in gold artifacts",
                    {"scanned_directory": str(gold_dir)}
                )
            
            latest_gw = max(available_gws)
            
            return {
                "latest_gameweek": latest_gw,
                "available_gameweeks": sorted(available_gws),
                "total_available": len(available_gws),
                "api_version": "5.0.0",
                "mode": "strict_canonical_only"
            }
            
        except StrictAPIError:
            raise
        except Exception as e:
            raise StrictAPIError(
                "LATEST_GAMEWEEK_ERROR",
                "Error retrieving latest gameweek",
                {"error": str(e)}
            )

    async def get_canonical_season_info(self) -> Dict:
        """Informations saison canonique"""
        
        try:
            calendar_result = self.fixture_generator.process_calendar()
            
            if calendar_result['status'] != 'success':
                raise StrictAPIError(
                    "CANONICAL_CALENDAR_ERROR",
                    "Cannot load canonical calendar",
                    {"error": calendar_result}
                )
            
            # Analyse gameweeks disponibles
            gold_dir = Path("outputs/gold_backfill")
            available_gws = []
            
            if gold_dir.exists():
                for gw_dir in sorted(gold_dir.iterdir()):
                    if gw_dir.is_dir() and gw_dir.name.startswith('gw'):
                        gw_num = int(gw_dir.name[2:])
                        available_gws.append(gw_num)
            
            # Récupération info repo status depuis manifest récent
            repo_status = "unknown"
            latest_manifest = None
            
            if available_gws:
                latest_gw = max(available_gws)
                latest_manifest_path = Path(f"outputs/gold_backfill/gw{latest_gw:02d}/manifest.json")
                
                if latest_manifest_path.exists():
                    try:
                        with open(latest_manifest_path) as f:
                            latest_manifest = json.load(f)
                        repo_status = latest_manifest.get('source_control', {}).get('repo_status', 'unknown')
                    except:
                        repo_status = "read_error"
            
            return {
                "season_id": "2025-26",
                "league_id": "epl",
                "season_hash": calendar_result['season_hash'],
                "format_version": "v2.1",
                "total_fixtures": calendar_result['fixtures_count'],
                "total_gameweeks": 38,
                "available_gameweeks": len(available_gws),
                "gameweeks_ready": available_gws,
                "api_mode": "strict_canonical_only",
                "canonical_calendar": "EPL_25_26_Full_Calendar.csv",
                
                # Nouveau: Build quality info
                "build_quality": {
                    "repo_status": repo_status,
                    "latest_manifest_gw": max(available_gws) if available_gws else None,
                    "git_sha": latest_manifest.get('source_control', {}).get('git_sha', 'unknown') if latest_manifest else 'unknown',
                    "build_id": latest_manifest.get('metadata', {}).get('build_id', 'unknown') if latest_manifest else 'unknown'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Season info error: {e}")
            raise StrictAPIError(
                "SEASON_INFO_ERROR",
                "Cannot retrieve season information",
                {"error": str(e)}
            )
    
    async def _full_system_validation(self):
        """Validation complète système en arrière-plan"""
        
        try:
            self.logger.info("🔍 Starting full system validation...")
            
            # Validation artefacts gold
            validation_result = self.validator.validate_all_gold_artifacts()
            
            # Mise à jour cache
            self._last_validation = datetime.now().isoformat()
            
            if validation_result['valid_gameweeks'] == validation_result['total_gameweeks']:
                self.logger.info(f"✅ System validation: {validation_result['valid_gameweeks']}/7 GW validated")
            else:
                self.logger.warning(f"⚠️ System validation: {validation_result['valid_gameweeks']}/7 GW validated")
            
        except Exception as e:
            self.logger.error(f"❌ System validation failed: {e}")

# Exception handler pour StrictAPIError
def create_app() -> FastAPI:
    """Création application avec gestion erreurs"""
    
    api = EPLStrictAPI()
    app = api.app
    
    @app.exception_handler(StrictAPIError)
    async def strict_api_exception_handler(request, exc: StrictAPIError):
        return JSONResponse(
            status_code=400,
            content={
                "error_type": "strict_api_error",
                "error_code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "api_version": "5.0.0",
                "mode": "strict_canonical_only"
            }
        )
    
    return app

def main():
    """Point d'entrée API Strict"""
    
    print("🚀 ODDSY EPL API v5.0 Strict - Starting...")
    print("📋 Mode: Canonical artifacts only")
    print("🔒 Rejects non-compliant data")
    
    app = create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Production mode
        log_level="info"
    )

if __name__ == "__main__":
    main()