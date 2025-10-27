#!/usr/bin/env python3
"""
Cache Service pour API v5.3
===========================

Service de gestion des ETags et cache headers pour optimiser les performances
et garantir l'immutabilité des artefacts par gameweek.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from fastapi import Request
from fastapi.responses import Response


class CacheService:
    """Service de gestion du cache et des ETags"""
    
    def __init__(self):
        self.logger = logging.getLogger('CacheService')
        self._etag_cache: Dict[str, str] = {}  # Cache en mémoire des ETags
        
    def generate_etag(self, data: Any, include_keys: Optional[list] = None) -> str:
        """
        Génère un ETag fort basé sur le contenu des données
        
        Args:
            data: Données à hasher (dict, list, str)
            include_keys: Clés spécifiques à inclure dans le hash
            
        Returns:
            ETag sous forme "hash"
        """
        try:
            # Préparer les données pour le hash
            if isinstance(data, dict):
                if include_keys:
                    # Ne hasher que les clés spécifiées
                    hash_data = {k: data.get(k) for k in include_keys if k in data}
                else:
                    hash_data = data
                content = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
            elif isinstance(data, (list, tuple)):
                content = json.dumps(data, sort_keys=True, separators=(',', ':'))
            else:
                content = str(data)
            
            # Générer le hash SHA-256
            etag_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
            return f'"{etag_hash}"'
            
        except Exception as e:
            self.logger.error(f"Error generating ETag: {e}")
            # Fallback basé sur timestamp
            fallback_hash = hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]
            return f'"{fallback_hash}"'
    
    def generate_file_etag(self, file_path: Path) -> str:
        """
        Génère un ETag basé sur le fichier (taille + mtime)
        
        Args:
            file_path: Chemin vers le fichier
            
        Returns:
            ETag basé sur les métadonnées du fichier
        """
        try:
            if not file_path.exists():
                return '"not-found"'
            
            stat = file_path.stat()
            # Combiner taille et timestamp de modification
            content = f"{stat.st_size}-{stat.st_mtime}"
            etag_hash = hashlib.md5(content.encode()).hexdigest()[:12]
            return f'"{etag_hash}"'
            
        except Exception as e:
            self.logger.error(f"Error generating file ETag for {file_path}: {e}")
            return '"error"'
    
    def get_cache_headers_for_gameweek(self, gameweek: int, is_latest: bool = False) -> Dict[str, str]:
        """
        Génère les headers de cache appropriés pour une gameweek
        
        Args:
            gameweek: Numéro de gameweek
            is_latest: Si True, cache court pour latest, sinon cache long
            
        Returns:
            Dictionnaire des headers de cache
        """
        if is_latest:
            # Cache court pour /latest (5 minutes)
            max_age = 300
            headers = {
                "Cache-Control": f"public, max-age={max_age}, stale-while-revalidate=60",
                "Vary": "Accept, Accept-Encoding"
            }
        else:
            # Cache long pour /{gw} spécifique (24 heures)
            max_age = 86400
            headers = {
                "Cache-Control": f"public, max-age={max_age}, immutable",
                "Vary": "Accept, Accept-Encoding"
            }
        
        # Ajouter Expires pour compatibilité
        expires = datetime.utcnow() + timedelta(seconds=max_age)
        headers["Expires"] = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")
        
        return headers
    
    def check_if_none_match(self, request: Request, current_etag: str) -> bool:
        """
        Vérifie si le client a déjà la version courante (If-None-Match)
        
        Args:
            request: Requête FastAPI
            current_etag: ETag actuel des données
            
        Returns:
            True si le client a déjà la version courante
        """
        if_none_match = request.headers.get("If-None-Match")
        if not if_none_match:
            return False
        
        # Gérer les ETags multiples et wildcard
        client_etags = [etag.strip() for etag in if_none_match.split(",")]
        
        return current_etag in client_etags or "*" in client_etags
    
    def check_if_modified_since(self, request: Request, last_modified: datetime) -> bool:
        """
        Vérifie If-Modified-Since
        
        Args:
            request: Requête FastAPI
            last_modified: Timestamp de dernière modification
            
        Returns:
            True si modifié depuis la date demandée
        """
        if_modified_since = request.headers.get("If-Modified-Since")
        if not if_modified_since:
            return True
        
        try:
            # Parser la date HTTP
            client_time = datetime.strptime(if_modified_since, "%a, %d %b %Y %H:%M:%S GMT")
            # Comparer avec une précision à la seconde
            return last_modified.replace(microsecond=0) > client_time
        except ValueError:
            # Si on ne peut pas parser, on considère comme modifié
            return True
    
    def get_gameweek_metadata_for_etag(self, gameweek_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrait les métadonnées pertinentes pour l'ETag d'une gameweek
        
        Args:
            gameweek_data: Données complètes de la gameweek
            
        Returns:
            Métadonnées pour le calcul d'ETag
        """
        # Clés importantes pour l'immutabilité
        etag_keys = [
            "gameweek",
            "fixtures_count", 
            "predictions",
            "metadata.season_hash",
            "metadata.dataset_hash", 
            "metadata.git_sha",
            "metadata.generated_at"
        ]
        
        etag_data = {}
        for key in etag_keys:
            if "." in key:
                # Clé nested (e.g., "metadata.season_hash")
                parts = key.split(".")
                value = gameweek_data
                for part in parts:
                    value = value.get(part, {}) if isinstance(value, dict) else {}
                etag_data[key] = value
            else:
                etag_data[key] = gameweek_data.get(key)
        
        return etag_data
    
    def create_304_response(self, current_etag: str, cache_headers: Dict[str, str]) -> Response:
        """
        Crée une réponse 304 Not Modified
        
        Args:
            current_etag: ETag actuel
            cache_headers: Headers de cache
            
        Returns:
            Réponse 304 avec headers appropriés
        """
        headers = {
            "ETag": current_etag,
            **cache_headers
        }
        
        return Response(
            content="",
            status_code=304,
            headers=headers
        )
    
    def add_cache_headers_to_response(self, 
                                     response: Response, 
                                     etag: str, 
                                     cache_headers: Dict[str, str],
                                     last_modified: Optional[datetime] = None) -> Response:
        """
        Ajoute les headers de cache à une réponse
        
        Args:
            response: Réponse FastAPI
            etag: ETag à ajouter
            cache_headers: Headers de cache
            last_modified: Timestamp de dernière modification
            
        Returns:
            Réponse avec headers ajoutés
        """
        response.headers["ETag"] = etag
        
        for key, value in cache_headers.items():
            response.headers[key] = value
        
        if last_modified:
            response.headers["Last-Modified"] = last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT")
        
        # Headers de sécurité
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        return {
            "etag_cache_size": len(self._etag_cache),
            "etag_cache_keys": list(self._etag_cache.keys())
        }


# Instance globale du service cache
_cache_service: Optional[CacheService] = None

def get_cache_service() -> CacheService:
    """Récupère l'instance singleton du service cache"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service