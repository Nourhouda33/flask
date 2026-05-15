"""
Configuration centrale du projet Healthcare AI Platform.
Gère les variables d'environnement pour Flask, MySQL, JWT, Ollama, FAISS.
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration de base partagée par tous les environnements."""

    # ─── Flask ────────────────────────────────────────────────────────────────
    SECRET_KEY = os.getenv("SECRET_KEY", "healthcare-ai-secret-2024-change-in-prod")
    DEBUG = False
    TESTING = False

    # ─── MySQL / XAMPP ────────────────────────────────────────────────────────
    MYSQL_HOST     = os.getenv("MYSQL_HOST",     "localhost")
    MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER     = os.getenv("MYSQL_USER",     "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DB       = os.getenv("MYSQL_DB",       "healthcare_ai_platform")

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_POOL_RECYCLE        = 280
    SQLALCHEMY_POOL_TIMEOUT        = 20
    SQLALCHEMY_POOL_SIZE           = 10
    SQLALCHEMY_MAX_OVERFLOW        = 20

    # ─── JWT ──────────────────────────────────────────────────────────────────
    JWT_SECRET_KEY            = os.getenv("JWT_SECRET_KEY", "jwt-healthcare-secret-2024")
    JWT_ACCESS_TOKEN_EXPIRES  = timedelta(hours=int(os.getenv("JWT_ACCESS_HOURS",  "8")))
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=int(os.getenv("JWT_REFRESH_DAYS",  "30")))
    JWT_ALGORITHM             = "HS256"

    # ─── Ollama / LLM ─────────────────────────────────────────────────────────
    OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL",      "http://localhost:11434")
    OLLAMA_TIMEOUT       = int(os.getenv("OLLAMA_TIMEOUT",   "120"))

    # Modèles disponibles via Ollama
    LLAMA3_MODEL         = os.getenv("LLAMA3_MODEL",         "llama3")
    QWEN_CODER_MODEL     = os.getenv("QWEN_CODER_MODEL",     "qwen2.5-coder:7b-instruct")
    SQLCODER_MODEL       = os.getenv("SQLCODER_MODEL",       "sqlcoder")

    # ─── Embeddings biomédicaux ───────────────────────────────────────────────
    BIOBERT_MODEL        = os.getenv("BIOBERT_MODEL",        "dmis-lab/biobert-base-cased-v1.2")
    BIOLORD_MODEL        = os.getenv("BIOLORD_MODEL",        "FremyCompany/BioLORD-2023")
    EMBEDDING_DIMENSION  = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    # ─── FAISS ────────────────────────────────────────────────────────────────
    FAISS_INDEX_PATH     = os.getenv("FAISS_INDEX_PATH",     "embeddings/faiss_store/healthcare.index")
    FAISS_METADATA_PATH  = os.getenv("FAISS_METADATA_PATH",  "embeddings/faiss_store/metadata.json")

    # ─── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",")

    # ─── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL    = os.getenv("LOG_LEVEL",    "INFO")
    LOG_FILE     = os.getenv("LOG_FILE",     "logs/healthcare_ai.log")
    LOG_MAX_BYTES   = 10 * 1024 * 1024   # 10 MB
    LOG_BACKUP_COUNT = 5

    # ─── Pagination ───────────────────────────────────────────────────────────
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE     = 100

    # ─── Évaluation IA ────────────────────────────────────────────────────────
    EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "evaluation/datasets/medical_queries.json")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))


class DevelopmentConfig(Config):
    """Configuration pour l'environnement de développement."""
    DEBUG     = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Configuration pour l'environnement de production."""
    DEBUG     = False
    LOG_LEVEL = "WARNING"
    SQLALCHEMY_POOL_SIZE    = 20
    SQLALCHEMY_MAX_OVERFLOW = 40


class TestingConfig(Config):
    """Configuration pour les tests unitaires."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=5)


# Mapping nom → classe
config_map = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "testing":     TestingConfig,
    "default":     DevelopmentConfig,
}

def get_config() -> Config:
    """Retourne la configuration selon la variable d'environnement FLASK_ENV."""
    env = os.getenv("FLASK_ENV", "development")
    return config_map.get(env, DevelopmentConfig)
