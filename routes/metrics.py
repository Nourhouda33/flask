"""
Blueprint metrics — Métriques du pipeline IA et statut des modèles.

Endpoints :
  GET /api/metrics            → Métriques agrégées (EM, accuracy, latency...)
  GET /api/metrics/evaluation → Évaluation complète sur le dataset de référence
  GET /api/metrics/models     → Statut des modèles Ollama
"""

import time
import logging
from datetime import datetime, timedelta

from flask import Blueprint, request, g, current_app

from database.db import db
from models.ai_query_log import AIQueryLog
from auth.decorators import token_required, role_required
from utils.response_helper import success_response, error_response

logger      = logging.getLogger(__name__)
metrics_bp  = Blueprint("metrics", __name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_period(period: str) -> datetime | None:
    """Convertit une période en datetime de début."""
    now = datetime.utcnow()
    if period == "7d":
        return now - timedelta(days=7)
    if period == "30d":
        return now - timedelta(days=30)
    return None   # "all"


def _get_pipeline():
    """Récupère le pipeline IA depuis le contexte applicatif."""
    if not hasattr(current_app, "_ai_pipeline"):
        from ai.pipeline import create_pipeline
        current_app._ai_pipeline = create_pipeline(current_app._get_current_object())
    return current_app._ai_pipeline


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/metrics
# ─────────────────────────────────────────────────────────────────────────────

@metrics_bp.route("", methods=["GET"])
@token_required
@role_required("admin", "doctor")
def get_metrics():
    """
    Retourne les métriques agrégées du pipeline IA depuis AI_Query_Logs.

    Query params:
        period  (str) : '7d', '30d', 'all' (défaut 'all').
        user_id (int) : Filtrer par utilisateur (admin seulement).

    Returns:
        200 : {
            total_queries, exact_match_rate, avg_confidence,
            avg_latency_ms, latency_p50, latency_p90, latency_p99,
            intent_distribution, table_distribution,
            queries_per_day, period
        }
    """
    period  = request.args.get("period", "all")
    user_id = request.args.get("user_id", type=int)
    user    = g.current_user

    # Seul un admin peut voir les métriques d'un autre utilisateur
    if user_id and user_id != user.id_user and user.role != "admin":
        return error_response("Accès refusé", 403)

    # ── Construire la requête de base ──────────────────────────────────────
    query = AIQueryLog.query
    since = _parse_period(period)
    if since:
        query = query.filter(AIQueryLog.created_at >= since)
    if user_id:
        query = query.filter_by(user_id=user_id)

    logs = query.all()

    if not logs:
        return success_response(
            data={
                "total_queries":    0,
                "exact_match_rate": None,
                "avg_confidence":   None,
                "avg_latency_ms":   None,
                "latency_p50":      None,
                "latency_p90":      None,
                "latency_p99":      None,
                "intent_distribution": {},
                "table_distribution":  {},
                "queries_per_day":     {},
                "period":              period,
            },
            message="Aucune donnée disponible pour cette période",
        )

    # ── Calcul des métriques ───────────────────────────────────────────────
    total = len(logs)

    # Exact Match
    em_logs = [l for l in logs if l.exact_match is not None]
    em_rate = (
        sum(1 for l in em_logs if l.exact_match) / len(em_logs)
        if em_logs else None
    )

    # Confidence
    conf_logs = [l.confidence_score for l in logs if l.confidence_score is not None]
    avg_conf  = round(sum(conf_logs) / len(conf_logs), 4) if conf_logs else None

    # Latency
    lat_logs = sorted(l.latency_ms for l in logs if l.latency_ms is not None)
    avg_lat  = round(sum(lat_logs) / len(lat_logs), 1) if lat_logs else None

    def percentile(data, p):
        if not data:
            return None
        idx = min(int(len(data) * p / 100), len(data) - 1)
        return data[idx]

    # Distribution des intentions
    intent_dist: dict = {}
    for log in logs:
        intent = log.detected_intent or "unknown"
        intent_dist[intent] = intent_dist.get(intent, 0) + 1

    # Distribution des tables
    table_dist: dict = {}
    for log in logs:
        if log.detected_tables:
            tables = log.detected_tables if isinstance(log.detected_tables, list) else []
            for table in tables:
                table_dist[table] = table_dist.get(table, 0) + 1

    # Requêtes par jour (7 derniers jours)
    queries_per_day: dict = {}
    for log in logs:
        if log.created_at:
            day = log.created_at.strftime("%Y-%m-%d")
            queries_per_day[day] = queries_per_day.get(day, 0) + 1

    return success_response(
        data={
            "total_queries":       total,
            "exact_match_rate":    round(em_rate, 4) if em_rate is not None else None,
            "exact_match_count":   sum(1 for l in em_logs if l.exact_match),
            "avg_confidence":      avg_conf,
            "avg_latency_ms":      avg_lat,
            "latency_p50":         percentile(lat_logs, 50),
            "latency_p90":         percentile(lat_logs, 90),
            "latency_p99":         percentile(lat_logs, 99),
            "intent_distribution": intent_dist,
            "table_distribution":  table_dist,
            "queries_per_day":     queries_per_day,
            "period":              period,
            "computed_at":         datetime.utcnow().isoformat(),
        },
        message=f"Métriques calculées sur {total} requête(s)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/metrics/evaluation
# ─────────────────────────────────────────────────────────────────────────────

@metrics_bp.route("/evaluation", methods=["GET"])
@token_required
@role_required("admin")
def run_evaluation():
    """
    Lance l'évaluation complète du pipeline sur le dataset de référence (20 paires).
    Opération longue (~2-5 min selon les modèles Ollama).
    Réservé aux administrateurs.

    Query params:
        intent_only (bool) : Évaluer uniquement l'IntentAgent (plus rapide).

    Returns:
        200 : Rapport d'évaluation complet (EvaluationReport.to_dict())
        503 : Ollama indisponible.
    """
    intent_only = request.args.get("intent_only", "false").lower() == "true"

    pipeline = _get_pipeline()
    start    = time.perf_counter()

    try:
        from evaluation.metrics import MetricsCalculator, REFERENCE_DATASET

        calc = MetricsCalculator()

        if intent_only:
            # Évaluation rapide — IntentAgent uniquement
            result = calc.evaluate_intent_only(
                intent_agent=pipeline.intent_agent,
                dataset=REFERENCE_DATASET,
            )
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            return success_response(
                data={**result, "evaluation_time_ms": elapsed, "mode": "intent_only"},
                message=f"Évaluation IntentAgent terminée en {elapsed}ms",
            )
        else:
            # Évaluation complète du pipeline
            report  = calc.evaluate_pipeline(
                pipeline=pipeline,
                dataset=REFERENCE_DATASET,
                user_role="admin",
                verbose=False,
            )
            elapsed = round((time.perf_counter() - start) * 1000, 1)

            report_dict = report.to_dict()
            report_dict["evaluation_time_ms"] = elapsed
            report_dict["summary"]            = report.summary()

            return success_response(
                data=report_dict,
                message=f"Évaluation complète terminée — EM={report.exact_match_rate:.1%}",
            )

    except Exception as exc:
        logger.error("Erreur évaluation — %s", str(exc), exc_info=True)
        if "Ollama" in str(exc) or "Connection" in str(exc):
            return error_response(
                "Ollama indisponible. L'évaluation nécessite Llama3 et Qwen.",
                503,
                error="OLLAMA_UNAVAILABLE",
            )
        return error_response(f"Erreur lors de l'évaluation : {str(exc)[:300]}", 500)


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/metrics/models
# ─────────────────────────────────────────────────────────────────────────────

@metrics_bp.route("/models", methods=["GET"])
@token_required
@role_required("admin", "doctor")
def get_models_status():
    """
    Retourne le statut des modèles Ollama (Llama3, Qwen, SQLCoder).
    Teste chaque modèle avec une requête minimale et mesure le temps de réponse.

    Returns:
        200 : {
            ollama_available, models: {
                llama3:   { available, response_time_ms, error },
                qwen:     { available, response_time_ms, error },
                sqlcoder: { available, response_time_ms, error },
            },
            pipeline_status
        }
    """
    pipeline = _get_pipeline()
    client   = pipeline.client

    # ── Vérifier la disponibilité d'Ollama ────────────────────────────────
    ollama_available = client.is_available()

    models_status = {}

    if ollama_available:
        # Tester chaque modèle avec un prompt minimal
        test_configs = [
            ("llama3",   current_app.config.get("LLAMA3_MODEL",    "llama3")),
            ("qwen",     current_app.config.get("QWEN_CODER_MODEL","qwen2.5-coder:7b-instruct")),
            ("sqlcoder", current_app.config.get("SQLCODER_MODEL",  "sqlcoder")),
        ]

        for model_key, model_name in test_configs:
            start = time.perf_counter()
            try:
                response = client.generate(
                    model=model_name,
                    prompt="SELECT 1;",
                    system="Retourne uniquement: OK",
                    temperature=0.0,
                )
                elapsed = round((time.perf_counter() - start) * 1000, 1)
                models_status[model_key] = {
                    "model_name":       model_name,
                    "available":        True,
                    "response_time_ms": elapsed,
                    "response_preview": (response or "")[:50],
                    "error":            None,
                }
            except Exception as exc:
                elapsed = round((time.perf_counter() - start) * 1000, 1)
                models_status[model_key] = {
                    "model_name":       model_name,
                    "available":        False,
                    "response_time_ms": elapsed,
                    "response_preview": None,
                    "error":            str(exc)[:200],
                }
    else:
        # Ollama indisponible — tous les modèles sont down
        for key in ("llama3", "qwen", "sqlcoder"):
            models_status[key] = {
                "available": False,
                "error":     "Ollama indisponible",
            }

    # ── Statut du pipeline ─────────────────────────────────────────────────
    pipeline_status = pipeline.get_status()

    # ── Récupérer les modèles disponibles via l'API Ollama ─────────────────
    available_models = []
    if ollama_available:
        try:
            import requests as http_req
            resp = http_req.get(
                f"{client.base_url}/api/tags",
                timeout=5,
            )
            if resp.status_code == 200:
                tags_data = resp.json()
                available_models = [
                    {
                        "name":     m.get("name"),
                        "size_gb":  round(m.get("size", 0) / (1024 ** 3), 2),
                        "modified": m.get("modified_at", ""),
                    }
                    for m in tags_data.get("models", [])
                ]
        except Exception:
            pass

    return success_response(
        data={
            "ollama_available":  ollama_available,
            "ollama_url":        client.base_url,
            "models":            models_status,
            "available_models":  available_models,
            "pipeline_status":   pipeline_status,
            "checked_at":        datetime.utcnow().isoformat(),
        },
        message="Statut des modèles récupéré",
    )
