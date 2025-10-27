#!/usr/bin/env python3
"""
Configuration centralisée Backend FastAPI
==========================================
Settings avec chemins Pipeline Durci et sécurité
"""

from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    # API Configuration
    API_VERSION: str = "1.0"
    API_TITLE: str = "Oddsy Pipeline API"
    DEBUG: bool = False
    
    # Pipeline Durci Paths (absolus)
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    PIPELINE_PREDICTIONS_DIR: Path = PROJECT_ROOT / "predictions" / "production"
    PIPELINE_REPORTS_DIR: Path = PROJECT_ROOT / "reports" 
    PIPELINE_MODELS_DIR: Path = PROJECT_ROOT / "models" / "production"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    
    # Pipeline Scripts
    WEEKLY_AUTOMATION_SCRIPT: str = str(PROJECT_ROOT / "weekly_automation_real_data.py")
    TEMPORAL_CALCULATOR: str = str(PROJECT_ROOT / "enhanced_calculator_strict_temporal.py")
    
    # Security
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001", "https://oddsy.vercel.app"]
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    
    # Cache
    CACHE_TTL_PAST_ROUNDS: int = 86400  # 24h for historical
    CACHE_TTL_CURRENT_ROUND: int = 300   # 5min for current
    
    # Job Management  
    ENABLE_PIPELINE_TRIGGERS: bool = os.getenv("ENABLE_PIPELINE_TRIGGERS", "false").lower() == "true"
    MAX_CONCURRENT_JOBS: int = 1
    
    # Git version
    GIT_SHA: str = os.getenv("GIT_SHA", "unknown")
    
    # Football APIs for results
    FOOTBALL_DATA_API_KEY: str = ""
    API_FOOTBALL_KEY: str = ""
    FRONTEND_URL: str = "http://localhost:3000"
    REVALIDATION_SECRET: str = "dev_secret"
    
    class Config:
        env_file = ".env"

settings = Settings()