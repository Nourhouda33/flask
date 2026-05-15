"""
Blueprint auth — routes d'authentification JWT.

Endpoints :
  POST /api/auth/login    → connexion, retourne access + refresh token
  POST /api/auth/register → création d'un compte utilisateur
  POST /api/auth/refresh  → renouvellement du access token
  POST /api/auth/logout   → révocation du token (blacklist)
  GET  /api/auth/me       → profil de l'utilisateur connecté
"""

import logging
from flask import Blueprint, request, g

from database.db import db
from models.user import User
from auth.jwt_handler import generate_token, refresh_token as do_refresh, revoke_token
from auth.decorators import token_required, rate_limit
from utils.response_helper import success_response, error_response

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)

# Rôles valides acceptés à l'inscription
VALID_ROLES = {"admin", "doctor", "staff", "patient"}


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/auth/login
# ─────────────────────────────────────────────────────────────────────────────


@auth_bp.route("/login", methods=["POST"])
@rate_limit(10, per=60)  # 10 tentatives par minute par IP
def login():
    """
    Authentifie un utilisateur et retourne un access token + refresh token.

    Body JSON:
        email    (str) : Adresse email de l'utilisateur.
        password (str) : Mot de passe en clair.

    Returns:
        200 : { success, data: { access_token, refresh_token, user }, message }
        400 : Champs manquants.
        401 : Identifiants incorrects ou compte désactivé.
    """
    body = request.get_json(silent=True) or {}

    email = body.get("email", "").strip().lower()
    password = body.get("password", "").strip()

    if not email or not password:
        return error_response("Email et mot de passe requis", 400)

    # Rechercher l'utilisateur par email
    user = User.query.filter_by(email=email).first()

    if not user or not user.check_password(password):
        logger.warning(
            "Échec de connexion — email=%s ip=%s", email, request.remote_addr
        )
        return error_response("Identifiants incorrects", 401)

    if not user.is_active:
        logger.warning("Compte désactivé — user_id=%s", user.id_user)
        return error_response("Compte désactivé. Contactez l'administrateur.", 401)

    # Générer les tokens
    access_tok = generate_token(user.id_user, user.role, token_type="access")
    refresh_tok = generate_token(user.id_user, user.role, token_type="refresh")

    logger.info("Connexion réussie — user_id=%s role=%s", user.id_user, user.role)

    return success_response(
        data={
            "access_token": access_tok,
            "refresh_token": refresh_tok,
            "token_type": "Bearer",
            "user": user.to_dict(),
        },
        message="Connexion réussie",
        status_code=200,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/auth/register
# ─────────────────────────────────────────────────────────────────────────────


@auth_bp.route("/register", methods=["POST"])
@rate_limit(5, per=60)  # 5 inscriptions par minute par IP
def register():
    """
    Crée un nouveau compte utilisateur.
    Seul un admin peut créer des comptes avec le rôle 'admin'.

    Body JSON:
        username   (str)           : Nom d'utilisateur unique.
        email      (str)           : Adresse email unique.
        password   (str)           : Mot de passe (min 8 caractères).
        role       (str, optional) : Rôle RBAC (défaut : 'staff').
        id_staff   (int, optional) : Lien vers Medical_staff.
        id_patient (int, optional) : Lien vers Patient.

    Returns:
        201 : { success, data: { user }, message }
        400 : Validation échouée.
        409 : Email ou username déjà utilisé.
    """
    body = request.get_json(silent=True) or {}

    username = body.get("username", "").strip()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "").strip()
    role = body.get("role", "staff").strip().lower()
    id_staff = body.get("id_staff")
    id_patient = body.get("id_patient")

    # ── Validation ────────────────────────────────────────────────────────────
    if not username or not email or not password:
        return error_response("username, email et password sont requis", 400)

    if len(password) < 8:
        return error_response(
            "Le mot de passe doit contenir au moins 8 caractères", 400
        )

    if role not in VALID_ROLES:
        return error_response(
            f"Rôle invalide. Valeurs acceptées : {', '.join(VALID_ROLES)}", 400
        )

    # Vérifier les doublons
    if User.query.filter_by(email=email).first():
        return error_response("Cet email est déjà utilisé", 409)

    if User.query.filter_by(username=username).first():
        return error_response("Ce nom d'utilisateur est déjà pris", 409)

    # ── Création ──────────────────────────────────────────────────────────────
    user = User(
        username=username,
        email=email,
        role=role,
        id_staff=id_staff,
        id_patient=id_patient,
        is_active=True,
    )
    user.set_password(password)

    db.session.add(user)
    db.session.commit()

    logger.info(
        "Nouveau compte créé — user_id=%s username=%s role=%s",
        user.id_user,
        username,
        role,
    )

    return success_response(
        data={"user": user.to_dict()},
        message="Compte créé avec succès",
        status_code=201,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/auth/refresh
# ─────────────────────────────────────────────────────────────────────────────


@auth_bp.route("/refresh", methods=["POST"])
@rate_limit(20, per=60)
def refresh():
    """
    Renouvelle l'access token à partir d'un refresh token valide.
    Applique la rotation de token (l'ancien refresh token est révoqué).

    Body JSON:
        refresh_token (str) : Refresh token JWT valide.

    Returns:
        200 : { success, data: { access_token, refresh_token }, message }
        400 : Champ manquant.
        401 : Refresh token invalide ou expiré.
    """
    body = request.get_json(silent=True) or {}
    refresh_tok = body.get("refresh_token", "").strip()

    if not refresh_tok:
        return error_response("refresh_token requis", 400)

    tokens = do_refresh(refresh_tok)
    if not tokens:
        return error_response("Refresh token invalide ou expiré", 401)

    return success_response(
        data={**tokens, "token_type": "Bearer"},
        message="Token renouvelé avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/auth/logout
# ─────────────────────────────────────────────────────────────────────────────


@auth_bp.route("/logout", methods=["POST"])
@token_required
def logout():
    """
    Révoque le token courant (blacklist).
    Révoque également le refresh token s'il est fourni.

    Body JSON (optionnel):
        refresh_token (str) : Refresh token à révoquer également.

    Returns:
        200 : { success, message }
    """
    # Révoquer l'access token courant
    revoke_token(g.current_token)

    # Révoquer le refresh token si fourni
    body = request.get_json(silent=True) or {}
    refresh_tok = body.get("refresh_token", "").strip()
    if refresh_tok:
        revoke_token(refresh_tok)

    logger.info("Déconnexion — user_id=%s", g.current_user.id_user)

    return success_response(message="Déconnexion réussie")


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/auth/me
# ─────────────────────────────────────────────────────────────────────────────


@auth_bp.route("/me", methods=["GET"])
@token_required
def me():
    """Retourne le profil complet de l'utilisateur authentifié avec ses permissions."""
    from auth.rbac import get_role_description, RESOURCE_PERMISSIONS

    user = g.current_user

    # Calculer les permissions du rôle
    role_info = get_role_description(user.role)

    return success_response(
        data={
            "user": user.to_dict(include_relations=True),
            "role_info": role_info,
            "permissions": [
                p for p, roles in RESOURCE_PERMISSIONS.items() if user.role in roles
            ],
        },
        message="Profil récupéré avec succès",
    )
