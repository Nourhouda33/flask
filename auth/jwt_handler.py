"""
Gestionnaire JWT — génération, vérification, rafraîchissement et révocation.
Utilise PyJWT directement pour un contrôle total sur les tokens.
La blacklist est stockée en mémoire (set Python) — remplacer par Redis en production.
"""

import jwt
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from flask import current_app

logger = logging.getLogger(__name__)

# ── Blacklist en mémoire ───────────────────────────────────────────────────────
# En production, utiliser Redis : redis_client.setex(jti, ttl, "revoked")
_token_blacklist: set[str] = set()


def generate_token(user_id: int, role: str, token_type: str = "access") -> str:
    """
    Génère un JWT signé (access ou refresh).

    Args:
        user_id:    Identifiant de l'utilisateur (sub).
        role:       Rôle RBAC de l'utilisateur (admin/doctor/staff/patient).
        token_type: "access" (courte durée) ou "refresh" (longue durée).

    Returns:
        Token JWT encodé sous forme de chaîne.

    Raises:
        ValueError: Si token_type est invalide.
    """
    if token_type not in ("access", "refresh"):
        raise ValueError(f"token_type invalide : {token_type!r}. Attendu 'access' ou 'refresh'.")

    now = datetime.now(timezone.utc)

    if token_type == "access":
        expires_delta = current_app.config.get("JWT_ACCESS_TOKEN_EXPIRES", timedelta(hours=8))
    else:
        expires_delta = current_app.config.get("JWT_REFRESH_TOKEN_EXPIRES", timedelta(days=30))

    # Identifiant unique du token (pour la blacklist)
    import uuid
    jti = str(uuid.uuid4())

    payload = {
        "sub":  str(user_id),          # Subject — identifiant utilisateur
        "role": role,                   # Rôle RBAC
        "type": token_type,             # Type de token
        "jti":  jti,                    # JWT ID unique (révocation)
        "iat":  now,                    # Issued At
        "exp":  now + expires_delta,    # Expiration
    }

    token = jwt.encode(
        payload,
        current_app.config["JWT_SECRET_KEY"],
        algorithm=current_app.config.get("JWT_ALGORITHM", "HS256"),
    )

    logger.debug("Token %s généré pour user_id=%s role=%s jti=%s", token_type, user_id, role, jti)
    return token


def verify_token(token: str, expected_type: str = "access") -> Optional[dict]:
    """
    Vérifie et décode un JWT. Contrôle la blacklist et le type attendu.

    Args:
        token:         Token JWT à vérifier.
        expected_type: Type attendu ("access" ou "refresh").

    Returns:
        Payload décodé si valide, None sinon.
    """
    try:
        payload = jwt.decode(
            token,
            current_app.config["JWT_SECRET_KEY"],
            algorithms=[current_app.config.get("JWT_ALGORITHM", "HS256")],
        )

        # Vérifier que le token n'est pas révoqué
        jti = payload.get("jti")
        if jti and jti in _token_blacklist:
            logger.warning("Token révoqué utilisé — jti=%s", jti)
            return None

        # Vérifier le type de token
        if payload.get("type") != expected_type:
            logger.warning(
                "Type de token incorrect — attendu=%s reçu=%s",
                expected_type,
                payload.get("type"),
            )
            return None

        return payload

    except jwt.ExpiredSignatureError:
        logger.info("Token expiré")
        return None
    except jwt.InvalidTokenError as exc:
        logger.warning("Token invalide : %s", str(exc))
        return None


def refresh_token(refresh_tok: str) -> Optional[dict]:
    """
    Génère un nouveau access token à partir d'un refresh token valide.
    Révoque l'ancien refresh token (rotation).

    Args:
        refresh_tok: Refresh token JWT valide.

    Returns:
        Dictionnaire {"access_token": str, "refresh_token": str} ou None si invalide.
    """
    payload = verify_token(refresh_tok, expected_type="refresh")
    if not payload:
        return None

    user_id = int(payload["sub"])
    role    = payload["role"]

    # Révoquer l'ancien refresh token (rotation de token)
    revoke_token(refresh_tok)

    # Générer de nouveaux tokens
    new_access  = generate_token(user_id, role, token_type="access")
    new_refresh = generate_token(user_id, role, token_type="refresh")

    logger.info("Tokens rafraîchis pour user_id=%s", user_id)
    return {
        "access_token":  new_access,
        "refresh_token": new_refresh,
    }


def revoke_token(token: str) -> bool:
    """
    Ajoute le JTI du token à la blacklist (révocation).

    Args:
        token: Token JWT à révoquer.

    Returns:
        True si révoqué avec succès, False si le token est invalide.
    """
    try:
        # Décoder sans vérifier l'expiration pour pouvoir révoquer les tokens expirés
        payload = jwt.decode(
            token,
            current_app.config["JWT_SECRET_KEY"],
            algorithms=[current_app.config.get("JWT_ALGORITHM", "HS256")],
            options={"verify_exp": False},
        )
        jti = payload.get("jti")
        if jti:
            _token_blacklist.add(jti)
            logger.info("Token révoqué — jti=%s user_id=%s", jti, payload.get("sub"))
            return True
        return False
    except jwt.InvalidTokenError as exc:
        logger.warning("Impossible de révoquer le token : %s", str(exc))
        return False


def is_token_revoked(token: str) -> bool:
    """
    Vérifie si un token est dans la blacklist.

    Args:
        token: Token JWT à vérifier.

    Returns:
        True si révoqué, False sinon.
    """
    try:
        payload = jwt.decode(
            token,
            current_app.config["JWT_SECRET_KEY"],
            algorithms=[current_app.config.get("JWT_ALGORITHM", "HS256")],
            options={"verify_exp": False},
        )
        jti = payload.get("jti")
        return jti in _token_blacklist if jti else False
    except jwt.InvalidTokenError:
        return True  # Token invalide = considéré révoqué


def get_blacklist_size() -> int:
    """Retourne la taille actuelle de la blacklist (utile pour le monitoring)."""
    return len(_token_blacklist)
