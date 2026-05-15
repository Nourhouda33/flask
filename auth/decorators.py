"""
Décorateurs d'authentification et d'autorisation.
- @token_required     : vérifie le JWT et injecte l'utilisateur courant
- @role_required      : contrôle RBAC sur le rôle
- @rate_limit         : limitation de débit par IP (en mémoire)
"""

import time
import logging
from collections import defaultdict
from functools import wraps
from typing import Callable

from flask import request, g
from utils.response_helper import error_response

from auth.jwt_handler import verify_token

logger = logging.getLogger(__name__)

# ── Stockage rate-limit en mémoire ────────────────────────────────────────────
# Structure : { ip: [(timestamp, count), ...] }
# En production, utiliser Redis pour le partage entre workers.
_rate_limit_store: dict[str, list] = defaultdict(list)


# ─────────────────────────────────────────────────────────────────────────────
#  @token_required
# ─────────────────────────────────────────────────────────────────────────────

def token_required(f: Callable) -> Callable:
    """
    Décorateur qui vérifie le JWT dans l'en-tête Authorization.
    Injecte l'utilisateur courant dans flask.g.current_user.

    Usage:
        @token_required
        def my_route():
            user = g.current_user
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Extraire le token de l'en-tête Authorization: Bearer <token>
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return error_response("Token manquant ou format invalide", 401)

        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            return error_response("Token vide", 401)

        # Vérifier et décoder le token
        payload = verify_token(token, expected_type="access")
        if not payload:
            return error_response("Token invalide ou expiré", 401)

        # Charger l'utilisateur depuis la base de données
        from models.user import User
        user_id = int(payload["sub"])
        user = User.query.filter_by(id_user=user_id, is_active=True).first()

        if not user:
            return error_response("Utilisateur introuvable ou désactivé", 401)

        # Stocker dans le contexte de la requête Flask
        g.current_user  = user
        g.current_token = token
        g.token_payload = payload

        logger.debug("Authentifié — user_id=%s role=%s path=%s", user_id, user.role, request.path)
        return f(*args, **kwargs)

    return decorated


# ─────────────────────────────────────────────────────────────────────────────
#  @role_required
# ─────────────────────────────────────────────────────────────────────────────

def role_required(*allowed_roles: str) -> Callable:
    """
    Décorateur RBAC — vérifie que l'utilisateur possède l'un des rôles autorisés.
    Doit être utilisé APRÈS @token_required.

    Args:
        *allowed_roles: Rôles autorisés (ex: 'admin', 'doctor').

    Usage:
        @token_required
        @role_required('admin', 'doctor')
        def my_route():
            ...
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            # g.current_user est injecté par @token_required
            user = getattr(g, "current_user", None)
            if not user:
                return error_response("Authentification requise", 401)

            if user.role not in allowed_roles:
                logger.warning(
                    "Accès refusé — user_id=%s role=%s required=%s path=%s",
                    user.id_user, user.role, allowed_roles, request.path,
                )
                return error_response(
                    f"Accès refusé. Rôles autorisés : {', '.join(allowed_roles)}",
                    403,
                )

            return f(*args, **kwargs)
        return decorated
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
#  @rate_limit
# ─────────────────────────────────────────────────────────────────────────────

def rate_limit(max_requests: int, per: int = 60) -> Callable:
    """
    Décorateur de limitation de débit par adresse IP.
    Utilise une fenêtre glissante (sliding window).

    Args:
        max_requests: Nombre maximum de requêtes autorisées dans la fenêtre.
        per:          Durée de la fenêtre en secondes (défaut : 60s).

    Usage:
        @rate_limit(10, per=60)   # 10 requêtes par minute
        def my_route():
            ...
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            # Identifier le client par son IP (ou X-Forwarded-For derrière un proxy)
            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
            if client_ip:
                client_ip = client_ip.split(",")[0].strip()

            now = time.time()
            window_start = now - per

            # Nettoyer les entrées hors fenêtre
            _rate_limit_store[client_ip] = [
                ts for ts in _rate_limit_store[client_ip] if ts > window_start
            ]

            # Vérifier la limite
            request_count = len(_rate_limit_store[client_ip])
            if request_count >= max_requests:
                logger.warning(
                    "Rate limit dépassé — ip=%s count=%s max=%s path=%s",
                    client_ip, request_count, max_requests, request.path,
                )
                return error_response(
                    f"Trop de requêtes. Limite : {max_requests} par {per}s.",
                    429,
                )

            # Enregistrer la requête courante
            _rate_limit_store[client_ip].append(now)

            return f(*args, **kwargs)
        return decorated
    return decorator
