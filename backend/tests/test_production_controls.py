#!/usr/bin/env python3
"""
Tests unitaires pour les contrôles production API v5.3
=====================================================

Tests complets pour les nouveaux services de production:
- Rate limiting avec Redis
- Métriques Prometheus/OpenTelemetry  
- Validation de données renforcée
- Cache et ETag
- Atomic file operations
- Mode dégradé et revalidation
"""

import pytest
import asyncio
import json
import time
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, Request

# Import des services à tester
from services.production_metrics import ProductionMetricsService, get_metrics_service
from services.probability_validator import ProbabilityValidator
from services.coverage_validator import CoverageValidator
from services.atomic_writer import AtomicFileWriter
from services.cache_service import CacheService
from services.production_revalidation import ProductionRevalidationService
from middleware.production_rate_limiter import ProductionRateLimiter


class TestProductionMetrics:
    """Tests pour le service de métriques production"""
    
    def test_metrics_service_initialization(self):
        """Test initialisation du service de métriques"""
        metrics_service = ProductionMetricsService(
            service_name="test-service",
            enable_prometheus=False,  # Éviter les dépendances
            enable_otlp=False
        )
        
        assert metrics_service.service_name == "test-service"
        assert hasattr(metrics_service, 'logger')
    
    def test_metrics_disabled_gracefully(self):
        """Test que les métriques se désactivent proprement si deps manquantes"""
        with patch('services.production_metrics.PROMETHEUS_AVAILABLE', False):
            metrics_service = ProductionMetricsService()
            
            # Les méthodes ne doivent pas lever d'erreur
            metrics_service.record_api_request("GET", "/test", 200, 0.1)
            metrics_service.record_rate_limit_event("client1", "/api", "allowed")
            
            summary = metrics_service.get_metrics_summary()
            assert summary["enabled"] == False
    
    @pytest.mark.asyncio
    async def test_trace_operation_context_manager(self):
        """Test du context manager pour tracing"""
        metrics_service = ProductionMetricsService(enable_prometheus=False, enable_otlp=False)
        
        # Test opération réussie
        with metrics_service.trace_operation("test_operation", gameweek=8):
            pass  # Pas d'exception
        
        # Test opération avec erreur
        with pytest.raises(ValueError):
            with metrics_service.trace_operation("test_operation_error"):
                raise ValueError("Test error")


class TestProbabilityValidator:
    """Tests pour le validateur de probabilités"""
    
    def setup_method(self):
        self.validator = ProbabilityValidator()
    
    def test_clarke_normalization(self):
        """Test normalisation Clarke"""
        raw_probs = {"home": 0.5, "draw": 0.3, "away": 0.3}  # Sum = 1.1
        
        normalized = self.validator._clarke_normalization(raw_probs)
        
        # Vérifier que la somme fait 1.0
        total = sum(normalized.values())
        assert abs(total - 1.0) < 0.001
        
        # Vérifier que les proportions relatives sont préservées
        assert normalized["home"] > normalized["draw"]
        assert normalized["home"] > normalized["away"]
    
    def test_power_normalization(self):
        """Test normalisation Power"""
        raw_probs = {"home": 0.4, "draw": 0.2, "away": 0.2}  # Sum = 0.8
        
        normalized = self.validator._power_normalization(raw_probs)
        
        # Vérifier que la somme fait 1.0
        total = sum(normalized.values())
        assert abs(total - 1.0) < 0.001
    
    def test_validate_probability_constraints_valid(self):
        """Test validation avec probabilités valides"""
        valid_prediction = {
            "probabilities": {"home": 0.45, "draw": 0.30, "away": 0.25}
        }
        
        result = self.validator.validate_probability_constraints(valid_prediction)
        
        assert result["is_valid"] == True
        assert result["normalization_applied"] == False
    
    def test_validate_probability_constraints_invalid_sum(self):
        """Test validation avec somme incorrecte"""
        invalid_prediction = {
            "probabilities": {"home": 0.6, "draw": 0.3, "away": 0.3}  # Sum = 1.2
        }
        
        result = self.validator.validate_probability_constraints(invalid_prediction)
        
        assert result["is_valid"] == True  # Corrigé par normalisation
        assert result["normalization_applied"] == True
        assert result["method_used"] == "clarke"
        
        # Vérifier que les probabilités normalisées somment à 1
        normalized_sum = sum(result["normalized_probabilities"].values())
        assert abs(normalized_sum - 1.0) < 0.001
    
    def test_validate_negative_probabilities(self):
        """Test gestion des probabilités négatives"""
        invalid_prediction = {
            "probabilities": {"home": -0.1, "draw": 0.6, "away": 0.5}
        }
        
        result = self.validator.validate_probability_constraints(invalid_prediction)
        
        assert result["is_valid"] == False
        assert "negative" in result["error"].lower()


class TestCoverageValidator:
    """Tests pour le validateur de couverture"""
    
    def setup_method(self):
        self.validator = CoverageValidator()
    
    def test_valid_gameweek_coverage(self):
        """Test validation avec couverture correcte"""
        valid_gameweek = {
            "fixtures_count": 10,
            "predictions": {
                "Arsenal_vs_Chelsea": {"match_info": {"home": "Arsenal", "away": "Chelsea"}},
                "Liverpool_vs_Manchester_City": {"match_info": {"home": "Liverpool", "away": "Manchester_City"}},
                "Tottenham_vs_Brighton": {"match_info": {"home": "Tottenham", "away": "Brighton"}},
                "Aston_Villa_vs_Fulham": {"match_info": {"home": "Aston_Villa", "away": "Fulham"}},
                "Everton_vs_Newcastle": {"match_info": {"home": "Everton", "away": "Newcastle"}},
                "Brentford_vs_Crystal_Palace": {"match_info": {"home": "Brentford", "away": "Crystal_Palace"}},
                "Bournemouth_vs_West_Ham": {"match_info": {"home": "Bournemouth", "away": "West_Ham"}},
                "Nottingham_Forest_vs_Wolverhampton": {"match_info": {"home": "Nottingham_Forest", "away": "Wolverhampton"}},
                "Manchester_United_vs_Burnley": {"match_info": {"home": "Manchester_United", "away": "Burnley"}},
                "Sunderland_vs_Leeds_United": {"match_info": {"home": "Sunderland", "away": "Leeds_United"}}
            }
        }
        
        result = self.validator.validate_gameweek_coverage(valid_gameweek, 8)
        
        assert result["is_valid"] == True
        assert result["fixtures_count"] == 10
        assert len(result["errors"]) == 0
    
    def test_insufficient_fixtures(self):
        """Test avec nombre insuffisant de fixtures"""
        insufficient_gameweek = {
            "fixtures_count": 8,
            "predictions": {
                "Arsenal_vs_Chelsea": {"match_info": {"home": "Arsenal", "away": "Chelsea"}},
                "Liverpool_vs_Manchester_City": {"match_info": {"home": "Liverpool", "away": "Manchester_City"}}
            }
        }
        
        result = self.validator.validate_gameweek_coverage(insufficient_gameweek, 8)
        
        assert result["is_valid"] == False
        assert any("fixture count" in error.lower() for error in result["errors"])
    
    def test_non_epl_teams(self):
        """Test avec équipes non-EPL"""
        invalid_teams_gameweek = {
            "fixtures_count": 1,
            "predictions": {
                "Real_Madrid_vs_Barcelona": {"match_info": {"home": "Real_Madrid", "away": "Barcelona"}}
            }
        }
        
        result = self.validator.validate_gameweek_coverage(invalid_teams_gameweek, 8)
        
        assert result["is_valid"] == False
        assert any("non-EPL" in error for error in result["errors"])


class TestAtomicFileWriter:
    """Tests pour l'écriture atomique de fichiers"""
    
    def setup_method(self):
        self.writer = AtomicFileWriter()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        # Nettoyer les fichiers temporaires
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_atomic_json_write_success(self):
        """Test écriture atomique réussie"""
        test_data = {"test": "data", "gameweek": 8, "predictions": []}
        target_file = self.temp_dir / "test.json"
        
        self.writer.write_json_atomic(target_file, test_data)
        
        # Vérifier que le fichier existe et contient les bonnes données
        assert target_file.exists()
        
        with open(target_file) as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data
    
    def test_atomic_write_no_temp_files_left(self):
        """Test qu'aucun fichier temporaire ne reste après écriture"""
        test_data = {"test": "cleanup"}
        target_file = self.temp_dir / "cleanup_test.json"
        
        self.writer.write_json_atomic(target_file, test_data)
        
        # Vérifier qu'aucun fichier .tmp n'existe
        temp_files = list(self.temp_dir.glob("*.tmp"))
        assert len(temp_files) == 0
    
    def test_atomic_write_with_backup(self):
        """Test écriture avec backup du fichier existant"""
        target_file = self.temp_dir / "backup_test.json"
        
        # Créer un fichier initial
        initial_data = {"version": 1}
        with open(target_file, 'w') as f:
            json.dump(initial_data, f)
        
        # Écrire de nouvelles données avec backup
        new_data = {"version": 2}
        self.writer.write_json_atomic(target_file, new_data, create_backup=True)
        
        # Vérifier que le fichier a été mis à jour
        with open(target_file) as f:
            loaded_data = json.load(f)
        assert loaded_data == new_data
        
        # Vérifier que le backup existe
        backup_file = target_file.with_suffix('.json.backup')
        assert backup_file.exists()
        
        with open(backup_file) as f:
            backup_data = json.load(f)
        assert backup_data == initial_data
    
    def test_integrity_verification(self):
        """Test vérification d'intégrité des fichiers"""
        test_data = {"integrity": "test", "data": [1, 2, 3]}
        target_file = self.temp_dir / "integrity_test.json"
        
        # Écrire avec checksum
        self.writer.write_json_atomic(target_file, test_data, verify_integrity=True)
        
        # Vérifier que le fichier est valide
        is_valid = self.writer._verify_file_integrity(target_file, test_data)
        assert is_valid == True


class TestCacheService:
    """Tests pour le service de cache"""
    
    def setup_method(self):
        self.cache_service = CacheService()
    
    def test_etag_generation_consistent(self):
        """Test que l'ETag est consistant pour les mêmes données"""
        data = {"test": "data", "timestamp": "2025-01-01"}
        
        etag1 = self.cache_service.generate_etag(data)
        etag2 = self.cache_service.generate_etag(data)
        
        assert etag1 == etag2
        assert len(etag1) > 0
    
    def test_etag_different_for_different_data(self):
        """Test que l'ETag diffère pour des données différentes"""
        data1 = {"test": "data1"}
        data2 = {"test": "data2"}
        
        etag1 = self.cache_service.generate_etag(data1)
        etag2 = self.cache_service.generate_etag(data2)
        
        assert etag1 != etag2
    
    def test_cache_headers_immutable_gameweek(self):
        """Test headers de cache pour gameweek immutable"""
        headers = self.cache_service.get_cache_headers_for_gameweek(8, is_latest=False)
        
        assert "public" in headers["Cache-Control"]
        assert "max-age=86400" in headers["Cache-Control"]  # 24h
        assert "immutable" in headers["Cache-Control"]
    
    def test_cache_headers_latest_gameweek(self):
        """Test headers de cache pour latest gameweek"""
        headers = self.cache_service.get_cache_headers_for_gameweek(9, is_latest=True)
        
        assert "public" in headers["Cache-Control"]
        assert "max-age=300" in headers["Cache-Control"]  # 5min
        assert "immutable" not in headers["Cache-Control"]
    
    def test_if_none_match_check(self):
        """Test vérification If-None-Match"""
        data = {"test": "etag_check"}
        etag = self.cache_service.generate_etag(data)
        
        # Test correspondance exacte
        assert self.cache_service.check_if_none_match(f'"{etag}"', data) == True
        
        # Test pas de correspondance
        assert self.cache_service.check_if_none_match('"different-etag"', data) == False
        
        # Test wildcard
        assert self.cache_service.check_if_none_match("*", data) == True


class TestProductionRateLimiter:
    """Tests pour le rate limiter production"""
    
    def setup_method(self):
        # Mock Redis pour les tests
        self.mock_redis = Mock()
        self.rate_limiter = ProductionRateLimiter(
            redis_url="redis://mock",
            default_requests=10,
            default_window=60
        )
        self.rate_limiter.redis_client = self.mock_redis
    
    def test_rate_limit_key_generation(self):
        """Test génération des clés Redis"""
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        mock_request.url.path = "/api/v5/gameweeks/8/predictions"
        
        client_key = self.rate_limiter._get_client_key(mock_request)
        endpoint_pattern = self.rate_limiter._get_endpoint_pattern(mock_request.url.path)
        
        assert client_key.startswith("ip_")
        assert endpoint_pattern == "/api/v5/gameweeks/*/predictions"
    
    def test_rate_limit_pattern_matching(self):
        """Test matching des patterns d'endpoints"""
        test_cases = [
            ("/api/v5/gameweeks/8/predictions", "/api/v5/gameweeks/*/predictions"),
            ("/api/v1/health", "/api/v1/health"),
            ("/api/v1/ops/gameweeks/9/run", "/api/v1/ops/gameweeks/*/run"),
        ]
        
        for path, expected_pattern in test_cases:
            pattern = self.rate_limiter._get_endpoint_pattern(path)
            assert pattern == expected_pattern
    
    @pytest.mark.asyncio
    async def test_rate_limit_allowed_request(self):
        """Test requête autorisée sous la limite"""
        # Configurer le mock Redis
        self.mock_redis.pipeline.return_value.execute.return_value = [5]  # 5 requêtes actuelles
        
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        mock_request.url.path = "/api/v5/gameweeks/8/predictions"
        mock_request.state = Mock()
        
        result = await self.rate_limiter.check_rate_limit(mock_request)
        
        # Requête autorisée
        assert result is None
        assert hasattr(mock_request.state, 'rate_limit_headers')
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Test limite dépassée"""
        # Configurer le mock Redis pour limite dépassée
        self.mock_redis.pipeline.return_value.execute.return_value = [15]  # 15 > 10 (limite)
        
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        mock_request.url.path = "/api/v5/gameweeks/8/predictions"
        
        result = await self.rate_limiter.check_rate_limit(mock_request)
        
        # Vérifier que la réponse 429 est retournée
        assert result is not None
        assert result.status_code == 429
        assert "Rate limit exceeded" in result.body.decode()


class TestProductionRevalidation:
    """Tests pour le service de revalidation production"""
    
    def setup_method(self):
        self.revalidation_service = ProductionRevalidationService(
            frontend_url="http://localhost:3000",
            hmac_secret="test_secret_key"
        )
    
    def test_hmac_signature_generation(self):
        """Test génération signature HMAC"""
        payload = {
            "operation": "revalidate",
            "paths": ["/test"],
            "timestamp": 1700000000
        }
        timestamp = 1700000000
        
        signature = self.revalidation_service._generate_signature(payload, timestamp)
        
        assert len(signature) == 64  # SHA256 hex = 64 caractères
        assert signature.isalnum()
    
    def test_hmac_signature_consistency(self):
        """Test consistance des signatures HMAC"""
        payload = {"test": "data"}
        timestamp = 1700000000
        
        sig1 = self.revalidation_service._generate_signature(payload, timestamp)
        sig2 = self.revalidation_service._generate_signature(payload, timestamp)
        
        assert sig1 == sig2
    
    def test_signed_request_creation(self):
        """Test création de requête signée"""
        paths = ["/predictions/8", "/api/v5/gameweeks/8"]
        
        signed_request = self.revalidation_service._create_signed_request(paths)
        
        assert "payload" in signed_request
        assert "signature" in signed_request
        assert "timestamp" in signed_request
        
        payload = signed_request["payload"]
        assert payload["operation"] == "revalidate"
        assert payload["paths"] == paths
        assert payload["source"] == "oddsy-pipeline-api"
    
    @patch('requests.post')
    def test_revalidate_paths_success(self, mock_post):
        """Test revalidation réussie"""
        # Mock réponse réussie
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "revalidated": True,
            "paths": ["/test/path"]
        }
        mock_response.headers = {}
        mock_post.return_value = mock_response
        
        paths = ["/test/path"]
        result = self.revalidation_service.revalidate_paths(paths)
        
        assert result["success"] == True
        assert result["paths_revalidated"] == ["/test/path"]
        assert len(result["errors"]) == 0
    
    @patch('requests.post')
    def test_revalidate_paths_timeout(self, mock_post):
        """Test timeout de revalidation"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        paths = ["/test/path"]
        result = self.revalidation_service.revalidate_paths(paths)
        
        assert result["success"] == False
        assert any("timeout" in error["type"] for error in result["errors"])


# Tests d'intégration
class TestIntegrationProductionControls:
    """Tests d'intégration pour les contrôles production"""
    
    def test_full_pipeline_validation_flow(self):
        """Test du flow complet de validation"""
        # Simuler une gameweek complète
        gameweek_data = {
            "gameweek": 8,
            "fixtures_count": 10,
            "predictions": {}
        }
        
        # Ajouter 10 fixtures EPL valides
        epl_teams = [
            ("Arsenal", "Chelsea"), ("Liverpool", "Manchester_City"),
            ("Tottenham", "Brighton"), ("Aston_Villa", "Fulham"),
            ("Everton", "Newcastle"), ("Brentford", "Crystal_Palace"),
            ("Bournemouth", "West_Ham"), ("Nottingham_Forest", "Wolverhampton"),
            ("Manchester_United", "Burnley"), ("Sunderland", "Leeds_United")
        ]
        
        for i, (home, away) in enumerate(epl_teams):
            match_key = f"{home}_vs_{away}"
            gameweek_data["predictions"][match_key] = {
                "prediction": "H",
                "confidence": 0.7,
                "probabilities": {"home": 0.5, "draw": 0.3, "away": 0.2},
                "match_info": {"home": home, "away": away}
            }
        
        # Test validation de couverture
        coverage_validator = CoverageValidator()
        coverage_result = coverage_validator.validate_gameweek_coverage(gameweek_data, 8)
        assert coverage_result["is_valid"] == True
        
        # Test validation des probabilités pour chaque prédiction
        prob_validator = ProbabilityValidator()
        for prediction in gameweek_data["predictions"].values():
            prob_result = prob_validator.validate_probability_constraints(prediction)
            assert prob_result["is_valid"] == True
    
    def test_metrics_integration_with_services(self):
        """Test intégration des métriques avec les services"""
        # Test que les services utilisent bien les métriques
        with patch('services.production_metrics.get_metrics_service') as mock_get_metrics:
            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics
            
            # Test avec rate limiter
            rate_limiter = ProductionRateLimiter()
            # Simuler une opération qui devrait enregistrer des métriques
            # (ceci nécessiterait un mock plus complet du rate limiter)
            
            # Test avec validator
            validator = ProbabilityValidator()
            test_prediction = {"probabilities": {"home": 0.5, "draw": 0.3, "away": 0.2}}
            validator.validate_probability_constraints(test_prediction)
            
            # Vérifier que les métriques ont été appelées
            # (en production, on vérifierait les appels spécifiques)
            assert mock_get_metrics.called


if __name__ == "__main__":
    # Lancer les tests
    pytest.main([__file__, "-v"])