"""
Logger structuré pour Healthcare AI Platform.
- Format JSON pour la production (parsing facile par ELK/Datadog).
- Format lisible pour le développement.
- Middleware Flask pour logger chaque requête avec user_id, endpoint, durée.
"""

import os
import time
import json
import logging
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional

from flask import Flask, request, g


# ─────────────────────────────────────────────────────────────────────────────
#  Formatter JSON structuré
# ─────────────────────────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """
    Formatte les logs en JSON sur une seule ligne.
    Facilite l'ingestion par des outils comme ELK, Datadog, CloudWatch.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "module":    record.module,
            "function":  record.funcName,
            "line":      record.lineno,
        }

        # Ajouter les champs extra si présents (user_id, endpoint, etc.)
        for key in ("user_id", "endpoint", "method", "status_code", "duration_ms", "ip"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        # Ajouter la stack trace si exception
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Formatter lisible (développement)
# ─────────────────────────────────────────────────────────────────────────────

class ReadableFormatter(logging.Formatter):
    """Formatter lisible pour le développement avec couleurs ANSI."""

    COLORS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Vert
        "WARNING":  "\033[33m",   # Jaune
        "ERROR":    "\033[31m",   # Rouge
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:<8}{self.RESET}"

        # Champs extra optionnels
        extra = ""
        if hasattr(record, "user_id"):
            extra += f" | user={record.user_id}"
        if hasattr(record, "duration_ms"):
            extra += f" | {record.duration_ms}ms"
        if hasattr(record, "status_code"):
            extra += f" | HTTP {record.status_code}"

        base = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{level} | "
            f"{record.name:<35} | "
            f"{record.getMessage()}"
            f"{extra}"
        )

        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)

        return base


# ─────────────────────────────────────────────────────────────────────────────
#  setup_logger — initialisation principale
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(app: Flask) -> None:
    """
    Configure le système de logging de l'application Flask.

    - Développement : format lisible avec couleurs sur la console.
    - Production    : format JSON sur console + fichier rotatif.

    Args:
        app: Instance Flask.
    """
    log_level_str  = app.config.get("LOG_LEVEL", "INFO").upper()
    log_level      = getattr(logging, log_level_str, logging.INFO)
    log_file       = app.config.get("LOG_FILE",         "logs/healthcare_ai.log")
    max_bytes      = app.config.get("LOG_MAX_BYTES",     10 * 1024 * 1024)
    backup_count   = app.config.get("LOG_BACKUP_COUNT",  5)
    is_development = app.config.get("DEBUG", False)

    # Choisir le formatter selon l'environnement
    formatter = ReadableFormatter() if is_development else JSONFormatter()

    # ── Handler console ───────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # ── Handler fichier rotatif ───────────────────────────────────────────
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    # Le fichier utilise toujours JSON pour faciliter l'analyse
    file_handler.setLevel(log_level)
    file_handler.setFormatter(JSONFormatter())

    # ── Configuration du logger racine ────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    # Logger Flask
    app.logger.setLevel(log_level)
    if not app.logger.handlers:
        app.logger.addHandler(console_handler)
        app.logger.addHandler(file_handler)

    # Réduire le bruit des bibliothèques tierces
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if log_level_str == "DEBUG" else logging.WARNING
    )

    # ── Middleware de logging des requêtes ────────────────────────────────
    _register_request_logging(app)

    app.logger.info(
        "Logger initialisé",
        extra={"level": log_level_str, "file": log_file},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Middleware — log chaque requête HTTP
# ─────────────────────────────────────────────────────────────────────────────

def _register_request_logging(app: Flask) -> None:
    """
    Enregistre des hooks before/after_request pour logger chaque appel API.
    Chaque log inclut : user_id, endpoint, méthode, durée, status_code, IP.
    """
    api_logger = logging.getLogger("api.requests")

    @app.before_request
    def before_request_log():
        """Enregistre le timestamp de début de la requête."""
        g.request_start_time = time.perf_counter()

    @app.after_request
    def after_request_log(response):
        """
        Logue la requête terminée avec toutes les métriques.
        Exclut les requêtes vers /api/health pour éviter le bruit.
        """
        # Ignorer le health check
        if request.path == "/api/health":
            return response

        duration_ms = 0
        if hasattr(g, "request_start_time"):
            duration_ms = round((time.perf_counter() - g.request_start_time) * 1000, 2)

        # Récupérer l'utilisateur si authentifié
        user_id: Optional[int] = None
        if hasattr(g, "current_user") and g.current_user:
            user_id = g.current_user.id_user

        # IP client (derrière proxy)
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        if client_ip:
            client_ip = client_ip.split(",")[0].strip()

        # Niveau de log selon le status code
        status_code = response.status_code
        if status_code >= 500:
            log_fn = api_logger.error
        elif status_code >= 400:
            log_fn = api_logger.warning
        else:
            log_fn = api_logger.info

        log_fn(
            f"{request.method} {request.path} → {status_code}",
            extra={
                "user_id":     user_id,
                "endpoint":    request.path,
                "method":      request.method,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "ip":          client_ip,
            },
        )

        return response


# ─────────────────────────────────────────────────────────────────────────────
#  get_logger — helper pour les modules
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger nommé pour un module.
    Utilisation : logger = get_logger(__name__)

    Args:
        name: Nom du module (typiquement __name__).

    Returns:
        Instance logging.Logger configurée.
    """
    return logging.getLogger(name)
