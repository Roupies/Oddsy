#!/usr/bin/env python3
"""
Production Revalidation Service pour API v5.3
==============================================

Service de revalidation ISR sécurisé avec signatures HMAC
pour synchronisation frontend Next.js après publication.
"""

import os
import hmac
import hashlib
import time
import logging
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from services.cache_service import get_cache_service


class RevalidationError(Exception):
    """Exception spécialisée pour les erreurs de revalidation"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ProductionRevalidationService:
    """Service de revalidation production avec sécurité renforcée"""
    
    def __init__(self, 
                 frontend_url: str = None,
                 hmac_secret: str = None,
                 timeout: int = 30):
        """
        Initialise le service de revalidation
        
        Args:
            frontend_url: URL de base du frontend
            hmac_secret: Secret HMAC pour signer les requêtes
            timeout: Timeout des requêtes en secondes
        """
        self.frontend_url = frontend_url or os.getenv('FRONTEND_URL', 'http://localhost:3000')
        self.hmac_secret = hmac_secret or os.getenv('REVALIDATION_HMAC_SECRET')
        self.timeout = timeout
        self.logger = logging.getLogger('ProductionRevalidation')
        self.cache_service = get_cache_service()
        
        # Validation de la configuration
        if not self.hmac_secret:
            self.logger.warning("⚠️ REVALIDATION_HMAC_SECRET not configured - revalidation disabled")
            self.enabled = False
        else:
            self.enabled = True
            self.logger.info(f"✅ Production revalidation enabled for {self.frontend_url}")
    
    def _generate_signature(self, payload: Dict[str, Any], timestamp: int) -> str:
        """
        Génère une signature HMAC-SHA256 pour le payload
        
        Args:
            payload: Données à signer
            timestamp: Timestamp Unix
            
        Returns:
            Signature hexadécimale
        """
        if not self.hmac_secret:
            return ""
        
        # Créer la chaîne à signer
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        sign_string = f"{timestamp}:{payload_json}"
        
        # Générer HMAC-SHA256
        signature = hmac.new(
            self.hmac_secret.encode('utf-8'),
            sign_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _create_signed_request(self, paths: List[str], operation_id: str = None) -> Dict[str, Any]:
        """
        Crée une requête signée pour la revalidation
        
        Args:
            paths: Chemins à revalider
            operation_id: ID d'opération pour le tracking
            
        Returns:
            Payload signé prêt pour envoi
        """
        if not operation_id:
            operation_id = f"revalidate_{uuid.uuid4().hex[:12]}"
        
        timestamp = int(time.time())
        
        payload = {
            "operation": "revalidate",
            "operation_id": operation_id,
            "paths": paths,
            "timestamp": timestamp,
            "source": "oddsy-pipeline-api"
        }
        
        signature = self._generate_signature(payload, timestamp)
        
        return {
            "payload": payload,
            "signature": signature,
            "timestamp": timestamp
        }
    
    def _validate_response_signature(self, response_data: Dict[str, Any], response_signature: str) -> bool:
        """
        Valide la signature de réponse du frontend
        
        Args:
            response_data: Données de réponse
            response_signature: Signature reçue
            
        Returns:
            True si signature valide
        """
        if not self.hmac_secret or not response_signature:
            return False
        
        try:
            # Reconstruire la signature attendue
            response_json = json.dumps(response_data, sort_keys=True, separators=(',', ':'))
            expected_signature = hmac.new(
                self.hmac_secret.encode('utf-8'),
                response_json.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(expected_signature, response_signature)
            
        except Exception as e:
            self.logger.error(f"Error validating response signature: {e}")
            return False
    
    def revalidate_gameweek_paths(self, 
                                 gameweek: int,
                                 include_latest: bool = True,
                                 operation_id: str = None) -> Dict[str, Any]:
        """
        Revalide les chemins liés à une gameweek spécifique
        
        Args:
            gameweek: Numéro de gameweek
            include_latest: Si True, inclut also /latest
            operation_id: ID d'opération pour tracking
            
        Returns:
            Rapport de revalidation
        """
        if not self.enabled:
            return {
                "success": False,
                "reason": "revalidation_disabled",
                "message": "HMAC secret not configured"
            }
        
        # Construire la liste des chemins
        paths = [
            f"/predictions/{gameweek}",
            f"/matchday/{gameweek}",
            f"/api/v5/gameweeks/{gameweek}/predictions",
            f"/api/v5/gameweeks/{gameweek}/status"
        ]
        
        if include_latest:
            paths.extend([
                "/predictions/latest",
                "/matchday/latest", 
                "/api/v5/gameweeks/latest"
            ])
        
        return self.revalidate_paths(paths, operation_id)
    
    def revalidate_paths(self, paths: List[str], operation_id: str = None) -> Dict[str, Any]:
        """
        Revalide une liste de chemins spécifiques
        
        Args:
            paths: Liste des chemins à revalider
            operation_id: ID d'opération pour tracking
            
        Returns:
            Rapport de revalidation détaillé
        """
        if not self.enabled:
            return {
                "success": False,
                "reason": "revalidation_disabled",
                "message": "HMAC secret not configured",
                "paths": paths
            }
        
        start_time = datetime.utcnow()
        operation_id = operation_id or f"revalidate_{uuid.uuid4().hex[:12]}"
        
        report = {
            "success": False,
            "operation_id": operation_id,
            "paths_requested": paths,
            "paths_revalidated": [],
            "errors": [],
            "warnings": [],
            "timestamp": start_time.isoformat(),
            "duration_ms": 0,
            "frontend_url": self.frontend_url
        }
        
        try:
            # Créer la requête signée
            signed_request = self._create_signed_request(paths, operation_id)
            
            # Endpoint de revalidation
            revalidation_url = f"{self.frontend_url.rstrip('/')}/api/revalidate"
            
            # Headers sécurisés
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Oddsy-Pipeline-Revalidation/1.0",
                "X-Operation-ID": operation_id,
                "X-Signature": signed_request["signature"],
                "X-Timestamp": str(signed_request["timestamp"])
            }
            
            self.logger.info(f"🔄 Starting revalidation: {operation_id}, {len(paths)} paths")
            
            # Envoi de la requête
            response = requests.post(
                revalidation_url,
                json=signed_request["payload"],
                headers=headers,
                timeout=self.timeout
            )
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            report["duration_ms"] = round(duration_ms, 2)
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    
                    # Valider la signature de réponse si présente
                    response_signature = response.headers.get("X-Response-Signature")
                    if response_signature:
                        if not self._validate_response_signature(response_data, response_signature):
                            report["warnings"].append({
                                "type": "invalid_response_signature",
                                "message": "Response signature validation failed"
                            })
                    
                    # Analyser la réponse
                    if response_data.get("revalidated"):
                        report["success"] = True
                        report["paths_revalidated"] = response_data.get("paths", [])
                        
                        # Vérifier que tous les chemins ont été revalidés
                        missing_paths = set(paths) - set(report["paths_revalidated"])
                        if missing_paths:
                            report["warnings"].append({
                                "type": "partial_revalidation",
                                "message": f"Some paths not revalidated: {list(missing_paths)}",
                                "missing_paths": list(missing_paths)
                            })
                        
                        self.logger.info(
                            f"✅ Revalidation successful: {operation_id} "
                            f"({len(report['paths_revalidated'])}/{len(paths)} paths, {duration_ms:.1f}ms)"
                        )
                    else:
                        report["errors"].append({
                            "type": "revalidation_failed",
                            "message": "Frontend reported revalidation failure",
                            "response": response_data
                        })
                    
                    # Ajouter les erreurs de réponse si présentes
                    if response_data.get("errors"):
                        report["errors"].extend(response_data["errors"])
                    
                except json.JSONDecodeError as e:
                    report["errors"].append({
                        "type": "invalid_response_json",
                        "message": f"Cannot parse response JSON: {str(e)}",
                        "response_text": response.text[:500]
                    })
            else:
                report["errors"].append({
                    "type": "http_error",
                    "message": f"HTTP {response.status_code}: {response.reason}",
                    "status_code": response.status_code,
                    "response_text": response.text[:500]
                })
            
        except requests.exceptions.Timeout:
            report["errors"].append({
                "type": "timeout",
                "message": f"Request timeout after {self.timeout}s",
                "timeout_seconds": self.timeout
            })
        except requests.exceptions.ConnectionError as e:
            report["errors"].append({
                "type": "connection_error", 
                "message": f"Cannot connect to frontend: {str(e)}",
                "frontend_url": self.frontend_url
            })
        except Exception as e:
            report["errors"].append({
                "type": "unexpected_error",
                "message": f"Unexpected error: {str(e)}",
                "error_type": type(e).__name__
            })
        
        # Log des erreurs
        if report["errors"]:
            self.logger.error(
                f"❌ Revalidation failed: {operation_id} - "
                f"{len(report['errors'])} errors"
            )
            for error in report["errors"]:
                self.logger.error(f"  - {error['type']}: {error['message']}")
        
        return report
    
    def revalidate_after_publication(self, 
                                   gameweek: int,
                                   publication_type: str = "predictions") -> Dict[str, Any]:
        """
        Revalidation complète après publication d'une gameweek
        
        Args:
            gameweek: Gameweek publiée
            publication_type: Type de publication ("predictions", "results", etc.)
            
        Returns:
            Rapport de revalidation complet
        """
        operation_id = f"publish_gw{gameweek}_{publication_type}_{int(time.time())}"
        
        self.logger.info(
            f"🚀 Post-publication revalidation: GW{gameweek} ({publication_type})"
        )
        
        # Revalidation en 2 étapes pour optimiser
        reports = []
        
        # Étape 1: Revalider la gameweek spécifique
        gw_report = self.revalidate_gameweek_paths(
            gameweek, 
            include_latest=False,
            operation_id=f"{operation_id}_gw"
        )
        reports.append(("gameweek_specific", gw_report))
        
        # Étape 2: Revalider latest seulement si GW revalidation réussie
        if gw_report["success"]:
            latest_report = self.revalidate_paths(
                ["/predictions/latest", "/api/v5/gameweeks/latest"],
                operation_id=f"{operation_id}_latest"
            )
            reports.append(("latest_pages", latest_report))
        else:
            self.logger.warning("Skipping latest revalidation due to gameweek revalidation failure")
        
        # Consolider les rapports
        consolidated_report = {
            "operation_id": operation_id,
            "gameweek": gameweek,
            "publication_type": publication_type,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_success": all(report[1]["success"] for report in reports),
            "stages": {stage: report for stage, report in reports},
            "total_paths_revalidated": sum(len(report[1]["paths_revalidated"]) for _, report in reports),
            "total_errors": sum(len(report[1]["errors"]) for _, report in reports)
        }
        
        if consolidated_report["overall_success"]:
            self.logger.info(
                f"✅ Post-publication revalidation complete: GW{gameweek} "
                f"({consolidated_report['total_paths_revalidated']} paths)"
            )
        else:
            self.logger.error(
                f"❌ Post-publication revalidation failed: GW{gameweek} "
                f"({consolidated_report['total_errors']} errors)"
            )
        
        return consolidated_report
    
    def health_check(self) -> Dict[str, Any]:
        """
        Vérifie la santé du service de revalidation
        
        Returns:
            Status de santé complet
        """
        health = {
            "service": "production_revalidation",
            "enabled": self.enabled,
            "timestamp": datetime.utcnow().isoformat(),
            "configuration": {
                "frontend_url": self.frontend_url,
                "hmac_secret_configured": bool(self.hmac_secret),
                "timeout_seconds": self.timeout
            }
        }
        
        if not self.enabled:
            health["status"] = "disabled"
            health["message"] = "HMAC secret not configured"
            return health
        
        # Test de connectivité basique
        try:
            response = requests.get(
                f"{self.frontend_url}/api/health",
                timeout=5
            )
            health["frontend_reachable"] = response.status_code < 500
            health["frontend_status_code"] = response.status_code
        except Exception as e:
            health["frontend_reachable"] = False
            health["frontend_error"] = str(e)
        
        health["status"] = "healthy" if health.get("frontend_reachable", False) else "degraded"
        
        return health


# Instance globale du service
_revalidation_service: Optional[ProductionRevalidationService] = None

def get_revalidation_service() -> ProductionRevalidationService:
    """Récupère l'instance singleton du service de revalidation"""
    global _revalidation_service
    if _revalidation_service is None:
        _revalidation_service = ProductionRevalidationService()
    return _revalidation_service