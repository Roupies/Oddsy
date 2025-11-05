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
    PIPELINE_PREDICTIONS_DIR: Path = PROJECT_ROOT / "data"
    PIPELINE_REPORTS_DIR: Path = PROJECT_ROOT / "reports" 
    PIPELINE_MODELS_DIR: Path = PROJECT_ROOT / "models" / "production"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    
    # Pipeline Scripts (removed non-existent scripts for prod autosuffisance)
    
    # Security
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001", "https://oddsy.vercel.app"]
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    
    # Cache
    CACHE_TTL_PAST_ROUNDS: int = 86400  # 24h for historical
    CACHE_TTL_CURRENT_ROUND: int = 300   # 5min for current
    
    # Job Management  
    ENABLE_PIPELINE_TRIGGERS: bool = os.getenv("ENABLE_PIPELINE_TRIGGERS", "false").lower() == "true"
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
    
    # Git version
    GIT_SHA: str = os.getenv("GIT_SHA", "unknown")
    
    # External APIs (configured via .env)
    FOOTBALL_DATA_API_KEY: str = os.getenv("FOOTBALL_DATA_API_KEY", "")
    API_FOOTBALL_KEY: str = os.getenv("API_FOOTBALL_KEY", "")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    REVALIDATION_SECRET: str = os.getenv("REVALIDATION_SECRET", "dev_secret")
    
    # Odds APIs
    THE_ODDS_API_KEY: str = os.getenv("THE_ODDS_API_KEY", "")
    USE_REAL_ODDS_API: bool = os.getenv("USE_REAL_ODDS_API", "false").lower() == "true"
    
    class Config:
        env_file = "../.env"
        extra = "ignore"

settings = Settings()