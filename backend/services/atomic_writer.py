#!/usr/bin/env python3
"""
Atomic File Writer pour API v5.3
================================

Service d'écriture atomique pour garantir l'intégrité des artefacts
en cas de crash système ou interruption du processus.
"""

import os
import json
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import uuid


class AtomicWriteError(Exception):
    """Exception spécialisée pour les erreurs d'écriture atomique"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class AtomicFileWriter:
    """Service d'écriture atomique crash-safe pour artefacts critiques"""
    
    def __init__(self, verify_writes: bool = True):
        """
        Initialise le writer atomique
        
        Args:
            verify_writes: Si True, vérifie l'intégrité après écriture
        """
        self.verify_writes = verify_writes
        self.logger = logging.getLogger('AtomicFileWriter')
        
    def _generate_temp_path(self, target_path: Path) -> Path:
        """
        Génère un chemin temporaire pour l'écriture atomique
        
        Args:
            target_path: Chemin final du fichier
            
        Returns:
            Chemin temporaire unique
        """
        # Utiliser le même répertoire que le fichier final
        parent_dir = target_path.parent
        
        # Nom temporaire avec UUID pour éviter les collisions
        temp_name = f".{target_path.name}.tmp.{uuid.uuid4().hex[:8]}"
        return parent_dir / temp_name
    
    def _calculate_checksum(self, data: Union[str, bytes]) -> str:
        """
        Calcule le checksum SHA-256 des données
        
        Args:
            data: Données à hasher
            
        Returns:
            Checksum hexadécimal
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    def _fsync_file(self, file_path: Path) -> None:
        """
        Force la synchronisation du fichier sur disque
        
        Args:
            file_path: Chemin du fichier à synchroniser
            
        Raises:
            AtomicWriteError: Si fsync échoue
        """
        try:
            with open(file_path, 'r+b') as f:
                os.fsync(f.fileno())
        except Exception as e:
            raise AtomicWriteError(
                f"Failed to fsync file {file_path}: {str(e)}",
                {"file_path": str(file_path), "error": str(e)}
            )
    
    def _fsync_directory(self, dir_path: Path) -> None:
        """
        Force la synchronisation du répertoire (métadonnées)
        
        Args:
            dir_path: Chemin du répertoire à synchroniser
            
        Raises:
            AtomicWriteError: Si fsync du répertoire échoue
        """
        try:
            # Ouvrir le répertoire et forcer fsync (Unix/Linux)
            if os.name == 'posix':
                dir_fd = os.open(dir_path, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except Exception as e:
            # Pas critique sur tous les systèmes, mais on log
            self.logger.warning(f"Could not fsync directory {dir_path}: {e}")
    
    def _verify_file_integrity(self, file_path: Path, expected_checksum: str) -> bool:
        """
        Vérifie l'intégrité d'un fichier après écriture
        
        Args:
            file_path: Chemin du fichier à vérifier
            expected_checksum: Checksum attendu
            
        Returns:
            True si l'intégrité est correcte
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            actual_checksum = hashlib.sha256(content).hexdigest()
            return actual_checksum == expected_checksum
            
        except Exception as e:
            self.logger.error(f"Failed to verify file integrity {file_path}: {e}")
            return False
    
    def write_json_atomic(self, 
                         data: Dict[str, Any], 
                         target_path: Union[str, Path],
                         backup_existing: bool = True,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Écrit un fichier JSON de manière atomique
        
        Args:
            data: Données JSON à écrire
            target_path: Chemin de destination
            backup_existing: Si True, sauvegarde l'ancien fichier
            metadata: Métadonnées supplémentaires à inclure
            
        Returns:
            Rapport d'écriture avec métriques
            
        Raises:
            AtomicWriteError: Si l'écriture échoue
        """
        target_path = Path(target_path)
        write_report = {
            "success": False,
            "target_path": str(target_path),
            "timestamp": datetime.utcnow().isoformat(),
            "backup_created": False,
            "checksum": None,
            "size_bytes": 0,
            "duration_ms": None
        }
        
        start_time = datetime.utcnow()
        temp_path = None
        backup_path = None
        
        try:
            # Créer le répertoire parent si nécessaire
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ajouter les métadonnées au JSON si fournies
            if metadata:
                data = dict(data)  # Copie pour ne pas modifier l'original
                data.setdefault("_metadata", {}).update(metadata)
                data["_metadata"]["written_at"] = start_time.isoformat()
                data["_metadata"]["writer"] = "AtomicFileWriter"
            
            # Sérialiser en JSON avec indentation pour lisibilité
            json_content = json.dumps(data, indent=2, separators=(',', ': '), ensure_ascii=False)
            json_bytes = json_content.encode('utf-8')
            
            # Calculer le checksum
            checksum = self._calculate_checksum(json_bytes)
            write_report["checksum"] = checksum
            write_report["size_bytes"] = len(json_bytes)
            
            # Sauvegarder l'ancien fichier si demandé
            if backup_existing and target_path.exists():
                backup_path = target_path.with_suffix(f'.backup.{int(start_time.timestamp())}')
                try:
                    backup_path.write_bytes(target_path.read_bytes())
                    write_report["backup_created"] = True
                    write_report["backup_path"] = str(backup_path)
                except Exception as e:
                    self.logger.warning(f"Could not create backup {backup_path}: {e}")
            
            # Générer chemin temporaire
            temp_path = self._generate_temp_path(target_path)
            
            # Écriture atomique en 4 étapes:
            # 1. Écrire dans fichier temporaire
            with open(temp_path, 'wb') as f:
                f.write(json_bytes)
            
            # 2. Forcer sync du fichier temporaire
            self._fsync_file(temp_path)
            
            # 3. Renommer atomiquement (opération atomique au niveau OS)
            os.replace(temp_path, target_path)
            
            # 4. Forcer sync du répertoire (métadonnées)
            self._fsync_directory(target_path.parent)
            
            # Vérification d'intégrité si demandée
            if self.verify_writes:
                if not self._verify_file_integrity(target_path, checksum):
                    raise AtomicWriteError(
                        "File integrity verification failed after write",
                        {"target_path": str(target_path), "expected_checksum": checksum}
                    )
            
            # Calcul de la durée
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            write_report["duration_ms"] = round(duration, 2)
            write_report["success"] = True
            
            self.logger.info(
                f"Atomic write successful: {target_path} "
                f"({write_report['size_bytes']} bytes, {duration:.1f}ms)"
            )
            
            return write_report
            
        except Exception as e:
            # Nettoyage en cas d'erreur
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            
            error_details = {
                "target_path": str(target_path),
                "temp_path": str(temp_path) if temp_path else None,
                "backup_path": str(backup_path) if backup_path else None,
                "original_error": str(e)
            }
            
            raise AtomicWriteError(
                f"Atomic write failed for {target_path}: {str(e)}",
                error_details
            )
    
    def write_text_atomic(self, 
                         content: str, 
                         target_path: Union[str, Path],
                         encoding: str = 'utf-8',
                         backup_existing: bool = True) -> Dict[str, Any]:
        """
        Écrit un fichier texte de manière atomique
        
        Args:
            content: Contenu texte à écrire
            target_path: Chemin de destination
            encoding: Encodage du fichier
            backup_existing: Si True, sauvegarde l'ancien fichier
            
        Returns:
            Rapport d'écriture avec métriques
        """
        target_path = Path(target_path)
        start_time = datetime.utcnow()
        temp_path = None
        
        try:
            # Créer le répertoire parent si nécessaire
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Encoder le contenu
            content_bytes = content.encode(encoding)
            checksum = self._calculate_checksum(content_bytes)
            
            # Sauvegarder l'ancien fichier si demandé
            if backup_existing and target_path.exists():
                backup_path = target_path.with_suffix(f'.backup.{int(start_time.timestamp())}')
                try:
                    backup_path.write_bytes(target_path.read_bytes())
                except Exception as e:
                    self.logger.warning(f"Could not create backup {backup_path}: {e}")
            
            # Générer chemin temporaire
            temp_path = self._generate_temp_path(target_path)
            
            # Écriture atomique
            with open(temp_path, 'wb') as f:
                f.write(content_bytes)
            
            self._fsync_file(temp_path)
            os.replace(temp_path, target_path)
            self._fsync_directory(target_path.parent)
            
            # Vérification d'intégrité
            if self.verify_writes:
                if not self._verify_file_integrity(target_path, checksum):
                    raise AtomicWriteError(
                        "File integrity verification failed after write",
                        {"target_path": str(target_path), "expected_checksum": checksum}
                    )
            
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "target_path": str(target_path),
                "checksum": checksum,
                "size_bytes": len(content_bytes),
                "duration_ms": round(duration, 2),
                "encoding": encoding
            }
            
        except Exception as e:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            
            raise AtomicWriteError(
                f"Atomic text write failed for {target_path}: {str(e)}",
                {"target_path": str(target_path), "original_error": str(e)}
            )
    
    def cleanup_temp_files(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """
        Nettoie les fichiers temporaires orphelins
        
        Args:
            directory: Répertoire à nettoyer
            
        Returns:
            Rapport de nettoyage
        """
        directory = Path(directory)
        cleanup_report = {
            "directory": str(directory),
            "temp_files_found": 0,
            "temp_files_removed": 0,
            "errors": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Chercher les fichiers .tmp.*
            temp_files = list(directory.glob("*.tmp.*"))
            cleanup_report["temp_files_found"] = len(temp_files)
            
            for temp_file in temp_files:
                try:
                    # Vérifier que le fichier est ancien (>5 minutes)
                    age_seconds = datetime.utcnow().timestamp() - temp_file.stat().st_mtime
                    if age_seconds > 300:  # 5 minutes
                        temp_file.unlink()
                        cleanup_report["temp_files_removed"] += 1
                        self.logger.info(f"Removed orphaned temp file: {temp_file}")
                except Exception as e:
                    cleanup_report["errors"].append({
                        "file": str(temp_file),
                        "error": str(e)
                    })
            
        except Exception as e:
            cleanup_report["errors"].append({
                "type": "directory_scan_error",
                "error": str(e)
            })
        
        return cleanup_report
    
    def get_file_status(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Obtient le statut d'un fichier avec métriques
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            Statut avec métriques
        """
        file_path = Path(file_path)
        
        status = {
            "path": str(file_path),
            "exists": file_path.exists(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if file_path.exists():
            try:
                stat = file_path.stat()
                status.update({
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "permissions": oct(stat.st_mode)[-3:],
                    "is_file": file_path.is_file(),
                    "is_symlink": file_path.is_symlink()
                })
                
                # Calculer checksum si c'est un fichier
                if file_path.is_file() and stat.st_size < 100 * 1024 * 1024:  # <100MB
                    try:
                        content = file_path.read_bytes()
                        status["checksum"] = self._calculate_checksum(content)
                    except Exception as e:
                        status["checksum_error"] = str(e)
                        
            except Exception as e:
                status["stat_error"] = str(e)
        
        return status


# Instance globale du writer atomique
_atomic_writer: Optional[AtomicFileWriter] = None

def get_atomic_writer() -> AtomicFileWriter:
    """Récupère l'instance singleton du writer atomique"""
    global _atomic_writer
    if _atomic_writer is None:
        _atomic_writer = AtomicFileWriter()
    return _atomic_writer