#!/usr/bin/env python3
"""
Frontend Revalidation Service
============================

Service pour déclencher la revalidation ISR du frontend Next.js
après publication de nouvelles prédictions.
"""

import os
import logging
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime

class FrontendRevalidationService:
    """Service de revalidation frontend"""
    
    def __init__(self, 
                 frontend_url: str = None, 
                 revalidation_secret: str = None):
        """
        Initialise le service de revalidation
        
        Args:
            frontend_url: URL de base du frontend (ex: https://oddsy.com)
            revalidation_secret: Secret pour authentification revalidation
        """
        self.frontend_url = frontend_url or os.getenv('FRONTEND_URL', 'http://localhost:3000')
        self.revalidation_secret = revalidation_secret or os.getenv('REVALIDATION_SECRET')
        self.logger = self._setup_logging()
        
        # URL de l'endpoint revalidation
        self.revalidation_endpoint = f"{self.frontend_url.rstrip('/')}/api/revalidate"
        
    def _setup_logging(self) -> logging.Logger:
        """Configure le logging"""
        logger = logging.getLogger('FrontendRevalidation')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def revalidate_after_predictions(self, gameweek: int) -> Dict[str, Any]:
        """
        Revalide le frontend après publication de nouvelles prédictions
        
        Args:
            gameweek: Numéro de gameweek des nouvelles prédictions
            
        Returns:
            Dict avec status de revalidation
        """
        if not self.revalidation_secret:
            self.logger.warning("⚠️ REVALIDATION_SECRET not configured - skipping frontend revalidation")
            return {
                "revalidated": False,
                "reason": "REVALIDATION_SECRET not configured",
                "paths": []
            }
        
        # Paths à revalider après nouvelles prédictions
        paths_to_revalidate = [
            "/",  # Homepage
            "/predictions/latest",  # Latest predictions
            f"/predictions/{gameweek}",  # Specific gameweek
            "/api/v5/gameweeks",  # API cache
            f"/api/v5/gameweeks/{gameweek}",  # Specific gameweek API
        ]
        
        return self.revalidate_paths(paths_to_revalidate)
    
    def revalidate_paths(self, paths: List[str]) -> Dict[str, Any]:
        """
        Revalide des paths spécifiques
        
        Args:
            paths: Liste des paths à revalider
            
        Returns:
            Dict avec résultat de revalidation
        """
        try:
            payload = {
                "secret": self.revalidation_secret,
                "paths": paths
            }
            
            self.logger.info(f"🔄 Triggering frontend revalidation for {len(paths)} paths...")
            
            response = requests.post(
                self.revalidation_endpoint,
                json=payload,
                timeout=30,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Oddsy-Pipeline-Revalidation/1.0"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"✅ Frontend revalidation successful: {result.get('paths', [])}")
                if result.get('errors'):
                    self.logger.warning(f"⚠️ Some revalidation errors: {result['errors']}")
                return result
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self.logger.error(f"❌ Frontend revalidation failed: {error_msg}")
                return {
                    "revalidated": False,
                    "error": error_msg,
                    "paths": []
                }
                
        except requests.exceptions.Timeout:
            error_msg = "Revalidation request timed out"
            self.logger.error(f"❌ {error_msg}")
            return {
                "revalidated": False,
                "error": error_msg,
                "paths": []
            }
        except requests.exceptions.ConnectionError:
            error_msg = f"Cannot connect to frontend at {self.frontend_url}"
            self.logger.warning(f"⚠️ {error_msg}")
            return {
                "revalidated": False,
                "error": error_msg,
                "paths": []
            }
        except Exception as e:
            error_msg = f"Unexpected error during revalidation: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {
                "revalidated": False,
                "error": error_msg,
                "paths": []
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Vérifie la santé du service de revalidation
        
        Returns:
            Dict avec status de santé
        """
        try:
            # Test simple endpoint
            response = requests.get(
                f"{self.frontend_url}/api/revalidate?secret=healthcheck",
                timeout=10
            )
            
            return {
                "healthy": True,
                "frontend_reachable": True,
                "endpoint": self.revalidation_endpoint,
                "secret_configured": bool(self.revalidation_secret)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "frontend_reachable": False,
                "endpoint": self.revalidation_endpoint,
                "secret_configured": bool(self.revalidation_secret),
                "error": str(e)
            }

# Instance globale du service
_revalidation_service = None

def get_revalidation_service() -> FrontendRevalidationService:
    """Récupère l'instance singleton du service de revalidation"""
    global _revalidation_service
    if _revalidation_service is None:
        _revalidation_service = FrontendRevalidationService()
    return _revalidation_service