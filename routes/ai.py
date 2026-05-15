"""
Blueprint AI — Routes du pipeline Text2SQL Healthcare.

Endpoints :
  POST   /api/prompt                → Pipeline complet (intent → SQL ou form)
  POST   /api/generate-sql          → Génération SQL directe via Qwen
  POST   /api/detect-tables         → Détection de tables (FAISS + LLM)
  POST   /api/detect-attributes     → Détection d'attributs + champs manquants
  POST   /api/execute               → Exécution SQL sécurisée
  GET    /api/chat-history          → Historique des requêtes
  DELETE /api/chat-history/<log_id> → Suppression d'une entrée
  GET    /api/prompt/stream         → Pipeline en streaming (SSE)
"""

import re
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

from flask import Blueprint, request, g, Response, stream_with_context, current_app

from database.db import db
from models.ai_query_log import AIQueryLog
from auth.decorators import token_required, role_required, rate_limit
from utils.response_helper import success_response, error_response

logger = logging.getLogger(__name__)

ai_bp = Blueprint("ai", __name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Classification SQL : DML vs DDL
# ─────────────────────────────────────────────────────────────────────────────

# DML (Data Manipulation Language) — supporté selon le rôle
DML_STATEMENTS = {"SELECT", "INSERT", "UPDATE", "DELETE"}

# DDL (Data Definition Language) — bloqué par défaut, admin seulement si activé
DDL_STATEMENTS = {"CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME"}

# DCL (Data Control Language) — toujours bloqué (sécurité)
DCL_STATEMENTS = {"GRANT", "REVOKE"}

# ─────────────────────────────────────────────────────────────────────────────
#  Blacklist SQL — opérations toujours interdites (injection + DCL + DDL dangereux)
# ─────────────────────────────────────────────────────────────────────────────

# Ces patterns sont bloqués pour TOUS les rôles, sans exception
SQL_BLACKLIST_PATTERNS = [
    # DDL destructif — irréversible
    r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA|INDEX|VIEW)\b",
    r"\bTRUNCATE\b",
    # DDL structurel — réservé aux DBA, pas au chat IA
    r"\bALTER\s+(?:TABLE|DATABASE|SCHEMA)\b",
    r"\bCREATE\s+(?:DATABASE|SCHEMA)\b",  # Créer une DB entière = interdit
    # DCL — contrôle d'accès, toujours interdit
    r"\bGRANT\b",
    r"\bREVOKE\b",
    # Exfiltration de données
    r"\bINTO\s+OUTFILE\b",
    r"\bINTO\s+DUMPFILE\b",
    r"\bLOAD_FILE\b",
    r"\bINFORMATION_SCHEMA\b",
    r"\bSHOW\s+DATABASES\b",
    # Injection SQL classique
    r";\s*(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)\b",  # Requêtes multiples
    r"SLEEP\s*\(",  # Time-based blind injection
    r"BENCHMARK\s*\(",  # CPU-based injection
    r"WAITFOR\s+DELAY",  # SQL Server injection
    r"--\s*$",  # Commentaire de fin de ligne (injection)
    r"/\*.*?\*/",  # Commentaire bloc (injection)
]

# ─────────────────────────────────────────────────────────────────────────────
#  Rôles et permissions
# ─────────────────────────────────────────────────────────────────────────────

# DML écriture : INSERT, UPDATE
WRITE_ROLES = {"admin", "doctor", "staff"}
# DML suppression : DELETE
DELETE_ROLES = {"admin"}
# DDL limité (CREATE TABLE, CREATE INDEX, ALTER ADD COLUMN) — admin uniquement
DDL_ROLES = {"admin"}
# Rôles autorisés pour DELETE
DELETE_ROLES = {"admin"}


def _check_sql_blacklist(sql: str) -> Optional[str]:
    """
    Vérifie si le SQL contient des opérations interdites.

    Args:
        sql: Requête SQL à vérifier.

    Returns:
        Message d'erreur si interdit, None si autorisé.
    """
    sql_upper = sql.upper()
    for pattern in SQL_BLACKLIST_PATTERNS:
        if re.search(pattern, sql_upper, re.IGNORECASE | re.MULTILINE):
            return f"Opération SQL interdite détectée : pattern '{pattern}'"
    return None


def _get_pipeline():
    """
    Récupère ou crée l'instance du pipeline IA.
    Utilise le contexte applicatif Flask pour le singleton.
    """
    if not hasattr(current_app, "_ai_pipeline"):
        from ai.pipeline import create_pipeline

        current_app._ai_pipeline = create_pipeline(current_app._get_current_object())
    return current_app._ai_pipeline


def _detect_sql_action(sql: str) -> str:
    """Détecte l'action SQL principale (SELECT/INSERT/UPDATE/DELETE)."""
    upper = sql.strip().upper()
    for action in ("SELECT", "INSERT", "UPDATE", "DELETE"):
        if upper.startswith(action):
            return action
    return "SELECT"


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/prompt — Pipeline complet
# ─────────────────────────────────────────────────────────────────────────────


@ai_bp.route("/prompt", methods=["POST"])
@token_required
@rate_limit(30, per=60)
def process_prompt():
    """
    Lance le pipeline Text2SQL complet sur un prompt en langage naturel.

    Body JSON:
        prompt          (str, requis) : Requête médicale en langage naturel.
        conversation_id (str)         : ID de conversation (optionnel, pour le contexte).

    Returns:
        READ_ONLY  : { sql, tables, attributes, filters, confidence, steps, latency_ms }
        READ_WRITE : { requires_input: true, form_schema, missing_attrs } si champs manquants
        READ_WRITE : { sql, tables, ... } si tous les champs sont fournis
        400 : Prompt vide.
        403 : Accès refusé (rôle insuffisant).
        503 : Ollama indisponible.
    """
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "").strip()

    if not prompt:
        return error_response("Le champ 'prompt' est requis", 400)

    if len(prompt) > 2000:
        return error_response("Le prompt ne peut pas dépasser 2000 caractères", 400)

    user = g.current_user
    pipeline = _get_pipeline()

    try:
        result = pipeline.process(
            prompt=prompt,
            user_role=user.role,
            user_id=user.id_user,
        )
    except Exception as exc:
        logger.error("Erreur pipeline /prompt — %s", str(exc), exc_info=True)
        # Détecter les erreurs Ollama spécifiquement
        if "Ollama" in str(exc) or "ConnectionError" in type(exc).__name__:
            return error_response(
                "Service IA indisponible. Vérifiez qu'Ollama est démarré.",
                503,
                error="OLLAMA_UNAVAILABLE",
            )
        return error_response(f"Erreur interne du pipeline : {str(exc)[:200]}", 500)

    # ── Erreur pipeline ────────────────────────────────────────────────────
    if result.error:
        if "Accès refusé" in result.error:
            return error_response(result.error, 403, error="ACCESS_DENIED")
        return error_response(result.error, 400)

    # ── READ-WRITE avec champs manquants → retourner le formulaire ─────────
    if result.form_required:
        return success_response(
            data={
                "requires_input": True,
                "form_schema": result.form_schema,
                "missing_attrs": result.missing_attrs,
                "intent": result.intent,
                "action": result.action,
                "tables": result.tables,
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
            },
            message="Informations supplémentaires requises",
        )

    # ── Résultat SQL ───────────────────────────────────────────────────────
    return success_response(
        data={
            "sql": result.sql,
            "intent": result.intent,
            "action": result.action,
            "tables": result.tables,
            "attributes": result.attributes,
            "filters": result.filters,
            "confidence": result.confidence,
            "valid_sql": result.valid_sql,
            "sql_errors": result.sql_errors,
            "sql_warnings": result.sql_warnings,
            "steps": result.to_dict()["steps"],
            "latency_ms": result.latency_ms,
            "source": result.source,
        },
        message="Requête traitée avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/prompt/stream — Pipeline en streaming (SSE)
# ─────────────────────────────────────────────────────────────────────────────


@ai_bp.route("/prompt/stream", methods=["GET"])
@token_required
def stream_prompt():
    """
    Lance le pipeline en mode streaming via Server-Sent Events (SSE).
    Envoie les étapes du pipeline au fur et à mesure de leur complétion.

    Query params:
        prompt (str) : Requête médicale en langage naturel.

    Returns:
        text/event-stream avec les événements :
          - step    : { name, status, duration_ms, details }
          - result  : Résultat final du pipeline
          - error   : Erreur si le pipeline échoue
          - done    : Signal de fin
    """
    prompt = request.args.get("prompt", "").strip()
    if not prompt:
        return error_response("Le paramètre 'prompt' est requis", 400)

    user = g.current_user

    def generate_events():
        """Générateur d'événements SSE."""
        pipeline = _get_pipeline()

        def send_event(event_type: str, data: dict) -> str:
            return (
                f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            )

        try:
            # Étape 1 : Intent detection
            yield send_event("step", {"name": "intent_detection", "status": "running"})
            intent_info = pipeline.intent_agent.analyze(prompt)
            yield send_event(
                "step",
                {
                    "name": "intent_detection",
                    "status": "success",
                    "action": intent_info.get("action"),
                    "confidence": intent_info.get("confidence"),
                    "source": intent_info.get("source"),
                },
            )

            # Étape 2 : Table matching
            yield send_event("step", {"name": "table_matching", "status": "running"})
            tables = pipeline.table_matcher.match_tables(prompt, intent_info)
            yield send_event(
                "step",
                {
                    "name": "table_matching",
                    "status": "success",
                    "tables": tables,
                },
            )

            # Étape 3 : Schema context
            yield send_event("step", {"name": "schema_context", "status": "running"})
            schema_context = pipeline.table_matcher.get_schema_context(tables)
            yield send_event("step", {"name": "schema_context", "status": "success"})

            # Étape 4 : SQL generation
            yield send_event("step", {"name": "sql_generation", "status": "running"})
            sql = pipeline.sql_generator.generate(prompt, schema_context, intent_info)
            yield send_event("step", {"name": "sql_generation", "status": "success"})

            # Étape 5 : SQL validation
            yield send_event("step", {"name": "sql_validation", "status": "running"})
            validation = pipeline.sql_validator.validate_and_fix(sql, schema_context)
            yield send_event(
                "step",
                {
                    "name": "sql_validation",
                    "status": "success" if validation["valid"] else "error",
                    "valid": validation["valid"],
                },
            )

            # Résultat final
            yield send_event(
                "result",
                {
                    "sql": validation["fixed_sql"],
                    "intent": intent_info.get("intent"),
                    "action": intent_info.get("action"),
                    "tables": tables,
                    "confidence": intent_info.get("confidence"),
                    "valid_sql": validation["valid"],
                },
            )

        except Exception as exc:
            logger.error("Erreur SSE pipeline — %s", str(exc))
            yield send_event("error", {"message": str(exc)[:200]})

        finally:
            yield send_event("done", {"timestamp": datetime.utcnow().isoformat()})

    return Response(
        stream_with_context(generate_events()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/generate-sql — Génération SQL directe
# ─────────────────────────────────────────────────────────────────────────────


@ai_bp.route("/generate-sql", methods=["POST"])
@token_required
@rate_limit(20, per=60)
def generate_sql():
    """
    Génère du SQL directement via Qwen sans passer par le pipeline complet.
    Utile pour les cas où l'intent et les tables sont déjà connus.

    Body JSON:
        prompt         (str, requis) : Requête en langage naturel.
        tables         (list)        : Tables à utiliser (optionnel).
        schema_context (str)         : Contexte schéma personnalisé (optionnel).

    Returns:
        200 : { sql, confidence, generation_time_ms, valid, errors }
        400 : Prompt manquant.
        503 : Qwen indisponible.
    """
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "").strip()

    if not prompt:
        return error_response("Le champ 'prompt' est requis", 400)

    pipeline = _get_pipeline()
    start = time.perf_counter()

    # Construire le contexte schéma
    tables = body.get("tables") or ["Patient"]
    schema_context = body.get(
        "schema_context"
    ) or pipeline.table_matcher.get_schema_context(tables)

    # Intent minimal pour le générateur
    intent_info = {
        "action": "SELECT",
        "tables": tables,
        "attributes": [],
        "filters": [],
    }

    try:
        sql = pipeline.sql_generator.generate(prompt, schema_context, intent_info)
    except Exception as exc:
        logger.error("Erreur generate-sql — %s", str(exc))
        if "Ollama" in str(exc) or "Connection" in str(exc):
            return error_response(
                "Qwen indisponible. Vérifiez Ollama.", 503, error="QWEN_UNAVAILABLE"
            )
        return error_response(f"Erreur de génération SQL : {str(exc)[:200]}", 500)

    # Valider le SQL généré
    validation = pipeline.sql_validator.validate_and_fix(sql, schema_context)
    generation_time = round((time.perf_counter() - start) * 1000, 1)

    return success_response(
        data={
            "sql": validation["fixed_sql"],
            "valid": validation["valid"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "fixes": validation["fixes"],
            "generation_time_ms": generation_time,
            "tables_used": tables,
        },
        message="SQL généré avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/detect-tables — Détection de tables
# ─────────────────────────────────────────────────────────────────────────────


@ai_bp.route("/detect-tables", methods=["POST"])
@token_required
@rate_limit(30, per=60)
def detect_tables():
    """
    Détecte les tables pertinentes pour un prompt via FAISS + LLM.

    Body JSON:
        prompt (str, requis) : Requête en langage naturel.

    Returns:
        200 : { tables, scores, method, intent_tables, latency_ms }
        400 : Prompt manquant.
    """
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "").strip()

    if not prompt:
        return error_response("Le champ 'prompt' est requis", 400)

    pipeline = _get_pipeline()
    start = time.perf_counter()

    # Détection via IntentAgent
    intent_info = pipeline.intent_agent.analyze(prompt)

    # Détection détaillée via TableMatcher
    detailed = pipeline.table_matcher.match_tables_detailed(prompt, intent_info)
    tables = [r["table"] for r in detailed]

    # Déterminer la méthode utilisée
    faiss_built = pipeline.table_matcher.faiss_index.is_built
    method = "hybrid" if faiss_built else "llama"

    latency = round((time.perf_counter() - start) * 1000, 1)

    return success_response(
        data={
            "tables": tables,
            "scores": detailed,
            "method": method,
            "intent_tables": intent_info.get("tables", []),
            "intent_source": intent_info.get("source", "unknown"),
            "latency_ms": latency,
        },
        message=f"{len(tables)} table(s) détectée(s)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/detect-attributes — Détection d'attributs
# ─────────────────────────────────────────────────────────────────────────────


@ai_bp.route("/detect-attributes", methods=["POST"])
@token_required
@rate_limit(30, per=60)
def detect_attributes():
    """
    Détecte les attributs nécessaires et les champs manquants pour un prompt.

    Body JSON:
        prompt (str, requis) : Requête en langage naturel.
        tables (list)        : Tables concernées (optionnel, auto-détecté si absent).

    Returns:
        200 : { attributes, missing, form_schema, action, requires_input }
        400 : Prompt manquant.
    """
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "").strip()

    if not prompt:
        return error_response("Le champ 'prompt' est requis", 400)

    pipeline = _get_pipeline()

    # Analyser l'intention
    intent_info = pipeline.intent_agent.analyze(prompt)

    # Surcharger les tables si fournies
    if body.get("tables"):
        intent_info["tables"] = body["tables"]

    # Détecter les attributs manquants
    missing = pipeline.missing_detector.detect(intent_info)
    form_schema = (
        pipeline.missing_detector.generate_form_schema(missing) if missing else None
    )
    req_missing = [m for m in missing if m["required"]]

    return success_response(
        data={
            "attributes": intent_info.get("attributes", []),
            "filters": intent_info.get("filters", []),
            "missing": missing,
            "form_schema": form_schema,
            "action": intent_info.get("action"),
            "tables": intent_info.get("tables", []),
            "requires_input": len(req_missing) > 0,
            "required_count": len(req_missing),
            "optional_count": len(missing) - len(req_missing),
        },
        message=f"{len(missing)} attribut(s) manquant(s) détecté(s)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/execute — Exécution SQL sécurisée
# ─────────────────────────────────────────────────────────────────────────────


@ai_bp.route("/execute", methods=["POST"])
@token_required
@rate_limit(20, per=60)
def execute_sql():
    """
    Exécute une requête SQL avec contrôle d'accès RBAC et protection blacklist.

    Body JSON:
        sql     (str, requis) : Requête SQL à exécuter.
        user_id (int)         : ID utilisateur pour le logging (optionnel).
        confirm (bool)        : Confirmation explicite pour les opérations WRITE.

    Protections :
        - Blacklist : DROP, TRUNCATE, ALTER, CREATE, GRANT, REVOKE interdits
        - RBAC : SELECT = tous, INSERT/UPDATE = doctor/staff/admin, DELETE = admin
        - Confirmation requise pour les opérations WRITE

    Returns:
        200 : { results, columns, row_count, execution_time_ms, action }
        400 : SQL invalide ou blacklisté.
        403 : Accès refusé.
        500 : Erreur d'exécution SQL.
    """
    body = request.get_json(silent=True) or {}
    sql = body.get("sql", "").strip()

    if not sql:
        return error_response("Le champ 'sql' est requis", 400)

    user = g.current_user
    confirm = body.get("confirm", False)

    # ── Vérification blacklist ─────────────────────────────────────────────
    blacklist_error = _check_sql_blacklist(sql)
    if blacklist_error:
        logger.warning(
            "SQL blacklisté — user_id=%s sql=%s error=%s",
            user.id_user,
            sql[:100],
            blacklist_error,
        )
        return error_response(blacklist_error, 400, error="SQL_BLACKLISTED")

    # ── Détection de l'action SQL ──────────────────────────────────────────
    action = _detect_sql_action(sql)

    # ── Contrôle d'accès RBAC centralisé ──────────────────────────────────
    from auth.rbac import check_sql_access

    access_error = check_sql_access(user.role, action)
    if access_error:
        return error_response(access_error, 403, error="INSUFFICIENT_ROLE")

    # ── Confirmation requise pour les opérations WRITE ─────────────────────
    if action in ("INSERT", "UPDATE", "DELETE") and not confirm:
        return success_response(
            data={
                "requires_confirmation": True,
                "action": action,
                "sql_preview": sql[:500],
                "message": f"Confirmez l'opération {action} en ajoutant confirm=true",
            },
            message=f"Confirmation requise pour {action}",
            status_code=200,
        )

    # ── Exécution SQL ──────────────────────────────────────────────────────
    start = time.perf_counter()
    try:
        from sqlalchemy import text

        result_proxy = db.session.execute(text(sql))

        if action == "SELECT":
            rows = result_proxy.fetchall()
            columns = list(result_proxy.keys()) if rows is not None else []
            results = [dict(zip(columns, row)) for row in rows]
            row_count = len(results)
        else:
            db.session.commit()
            results = []
            columns = []
            row_count = result_proxy.rowcount

        execution_time = round((time.perf_counter() - start) * 1000, 1)

        logger.info(
            "SQL exécuté — user_id=%s action=%s rows=%d time=%dms",
            user.id_user,
            action,
            row_count,
            execution_time,
        )

        # Sérialiser les résultats (convertir les types non-JSON)
        serialized_results = _serialize_results(results)

        return success_response(
            data={
                "results": serialized_results,
                "columns": columns,
                "row_count": row_count,
                "execution_time_ms": execution_time,
                "action": action,
            },
            message=f"Requête exécutée avec succès ({row_count} ligne(s))",
        )

    except Exception as exc:
        db.session.rollback()
        logger.error(
            "Erreur exécution SQL — user_id=%s error=%s sql=%s",
            user.id_user,
            str(exc),
            sql[:200],
        )
        return error_response(
            f"Erreur d'exécution SQL : {str(exc)[:300]}",
            500,
            error="SQL_EXECUTION_ERROR",
        )


def _serialize_results(results: list) -> list:
    """Sérialise les résultats SQL en types JSON-compatibles."""
    serialized = []
    for row in results:
        serialized_row = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                serialized_row[key] = value.isoformat()
            elif hasattr(value, "isoformat"):
                serialized_row[key] = value.isoformat()
            elif isinstance(value, bytes):
                serialized_row[key] = value.decode("utf-8", errors="replace")
            else:
                serialized_row[key] = value
        serialized.append(serialized_row)
    return serialized


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/chat-history — Historique des requêtes
# ─────────────────────────────────────────────────────────────────────────────


@ai_bp.route("/chat-history", methods=["GET"])
@token_required
def chat_history():
    """
    Retourne l'historique paginé des requêtes IA de l'utilisateur.

    Query params:
        user_id (int) : ID utilisateur (admin seulement, sinon user courant).
        limit   (int) : Nombre de résultats (défaut 50, max 200).
        offset  (int) : Décalage pour la pagination (défaut 0).
        period  (str) : Période : '7d', '30d', 'all' (défaut 'all').

    Returns:
        200 : { items: [...], total, limit, offset }
        403 : Accès refusé (consulter l'historique d'un autre utilisateur).
    """
    user = g.current_user

    # Déterminer l'user_id cible
    target_user_id = request.args.get("user_id", type=int)
    if target_user_id and target_user_id != user.id_user and user.role != "admin":
        return error_response(
            "Accès refusé : vous ne pouvez consulter que votre propre historique",
            403,
        )
    target_user_id = target_user_id or user.id_user

    # Paramètres de pagination
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = max(int(request.args.get("offset", 0)), 0)
    period = request.args.get("period", "all")

    # Construire la requête
    query = AIQueryLog.query.filter_by(user_id=target_user_id)

    # Filtre période
    if period == "7d":
        query = query.filter(
            AIQueryLog.created_at >= datetime.utcnow() - timedelta(days=7)
        )
    elif period == "30d":
        query = query.filter(
            AIQueryLog.created_at >= datetime.utcnow() - timedelta(days=30)
        )

    total = query.count()
    logs = (
        query.order_by(AIQueryLog.created_at.desc()).limit(limit).offset(offset).all()
    )

    return success_response(
        data={
            "items": [log.to_dict() for log in logs],
            "total": total,
            "limit": limit,
            "offset": offset,
            "period": period,
        },
        message=f"{total} requête(s) dans l'historique",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DELETE /api/chat-history/<log_id> — Suppression d'une entrée
# ─────────────────────────────────────────────────────────────────────────────


@ai_bp.route("/chat-history/<int:log_id>", methods=["DELETE"])
@token_required
def delete_chat_history(log_id: int):
    """
    Supprime une entrée de l'historique des requêtes IA.
    Un utilisateur ne peut supprimer que ses propres entrées (sauf admin).

    Returns:
        200 : { success, message }
        403 : Accès refusé.
        404 : Entrée introuvable.
    """
    user = g.current_user
    log = AIQueryLog.query.get(log_id)

    if not log:
        return error_response(f"Entrée d'historique {log_id} introuvable", 404)

    # Vérifier que l'utilisateur peut supprimer cette entrée
    if log.user_id != user.id_user and user.role != "admin":
        return error_response(
            "Accès refusé : vous ne pouvez supprimer que vos propres entrées",
            403,
        )

    try:
        db.session.delete(log)
        db.session.commit()
        logger.info(
            "Historique supprimé — log_id=%s par user_id=%s", log_id, user.id_user
        )
    except Exception as exc:
        db.session.rollback()
        return error_response(f"Erreur lors de la suppression : {str(exc)[:200]}", 500)

    return success_response(message=f"Entrée {log_id} supprimée de l'historique")
