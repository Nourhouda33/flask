"""
RBAC — Role-Based Access Control centralisé
Healthcare AI Platform

Matrice des permissions par rôle :

┌─────────────────────────────┬───────┬────────┬───────┬─────────┐
│ Permission                  │ admin │ doctor │ staff │ patient │
├─────────────────────────────┼───────┼────────┼───────┼─────────┤
│ SELECT (lecture)            │  ✅   │  ✅    │  ✅   │  ✅*   │
│ INSERT (création)           │  ✅   │  ✅    │  ✅   │  ❌    │
│ UPDATE (modification)       │  ✅   │  ✅    │  ✅   │  ❌    │
│ DELETE (suppression)        │  ✅   │  ❌    │  ❌   │  ❌    │
│ DDL (structure DB)          │  ✅   │  ❌    │  ❌   │  ❌    │
├─────────────────────────────┼───────┼────────┼───────┼─────────┤
│ Patients — lecture          │  ✅   │  ✅    │  ✅   │  ✅**  │
│ Patients — création/modif   │  ✅   │  ✅    │  ✅   │  ❌    │
│ Patients — suppression      │  ✅   │  ❌    │  ❌   │  ❌    │
├─────────────────────────────┼───────┼────────┼───────┼─────────┤
│ Dossier médical — lecture   │  ✅   │  ✅    │  ✅   │  ✅**  │
│ Dossier médical — modif     │  ✅   │  ✅    │  ❌   │  ❌    │
├─────────────────────────────┼───────┼────────┼───────┼─────────┤
│ Consultations — lecture     │  ✅   │  ✅    │  ✅   │  ✅**  │
│ Consultations — création    │  ✅   │  ✅    │  ✅   │  ❌    │
│ Consultations — modif       │  ✅   │  ✅    │  ❌   │  ❌    │
│ Consultations — suppression │  ✅   │  ❌    │  ❌   │  ❌    │
│ Rapport IA consultation     │  ✅   │  ✅    │  ❌   │  ❌    │
├─────────────────────────────┼───────┼────────┼───────┼─────────┤
│ Staff — lecture             │  ✅   │  ✅    │  ✅   │  ❌    │
│ Staff — CRUD                │  ✅   │  ❌    │  ❌   │  ❌    │
├─────────────────────────────┼───────┼────────┼───────┼─────────┤
│ Services — lecture          │  ✅   │  ✅    │  ✅   │  ❌    │
│ Services — CRUD             │  ✅   │  ❌    │  ❌   │  ❌    │
├─────────────────────────────┼───────┼────────┼───────┼─────────┤
│ AI Chat — SELECT prompts    │  ✅   │  ✅    │  ✅   │  ✅    │
│ AI Chat — WRITE prompts     │  ✅   │  ✅    │  ✅   │  ❌    │
│ AI Chat — DELETE prompts    │  ✅   │  ❌    │  ❌   │  ❌    │
│ AI Chat — DDL prompts       │  ✅   │  ❌    │  ❌   │  ❌    │
├─────────────────────────────┼───────┼────────┼───────┼─────────┤
│ Métriques IA                │  ✅   │  ✅    │  ❌   │  ❌    │
│ Évaluation dataset          │  ✅   │  ❌    │  ❌   │  ❌    │
│ Statut modèles Ollama       │  ✅   │  ✅    │  ❌   │  ❌    │
├─────────────────────────────┼───────┼────────┼───────┼─────────┤
│ Export PDF consultation     │  ✅   │  ✅    │  ✅   │  ❌    │
│ Export CSV/Excel            │  ✅   │  ✅    │  ✅   │  ❌    │
├─────────────────────────────┼───────┼────────┼───────┼─────────┤
│ Gestion utilisateurs        │  ✅   │  ❌    │  ❌   │  ❌    │
│ Historique IA (tous users)  │  ✅   │  ❌    │  ❌   │  ❌    │
└─────────────────────────────┴───────┴────────┴───────┴─────────┘

* patient : SELECT uniquement sur ses propres données (filtré par id_patient)
** patient : accès uniquement à son propre profil/dossier/consultations
"""

from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  Définition des rôles
# ─────────────────────────────────────────────────────────────────────────────

ROLE_ADMIN = "admin"
ROLE_DOCTOR = "doctor"
ROLE_STAFF = "staff"
ROLE_PATIENT = "patient"

ALL_ROLES = {ROLE_ADMIN, ROLE_DOCTOR, ROLE_STAFF, ROLE_PATIENT}
MEDICAL_ROLES = {ROLE_ADMIN, ROLE_DOCTOR, ROLE_STAFF}
PRIVILEGED_ROLES = {ROLE_ADMIN, ROLE_DOCTOR}

# ─────────────────────────────────────────────────────────────────────────────
#  Permissions SQL par rôle
# ─────────────────────────────────────────────────────────────────────────────

# Qui peut faire quoi en SQL via le pipeline IA
SQL_PERMISSIONS = {
    "SELECT": ALL_ROLES,  # Tous peuvent lire
    "INSERT": MEDICAL_ROLES,  # admin + doctor + staff
    "UPDATE": MEDICAL_ROLES,  # admin + doctor + staff
    "DELETE": {ROLE_ADMIN},  # admin uniquement
    "DDL": {ROLE_ADMIN},  # admin uniquement (CREATE TABLE, etc.)
}

# ─────────────────────────────────────────────────────────────────────────────
#  Permissions par ressource
# ─────────────────────────────────────────────────────────────────────────────

RESOURCE_PERMISSIONS = {
    # ── Patients ──────────────────────────────────────────────────────────
    "patients:read": ALL_ROLES,
    "patients:create": MEDICAL_ROLES,
    "patients:update": MEDICAL_ROLES,
    "patients:delete": {ROLE_ADMIN},
    # ── Dossier médical ───────────────────────────────────────────────────
    "medical_record:read": ALL_ROLES,
    "medical_record:update": PRIVILEGED_ROLES,  # admin + doctor seulement
    # ── Consultations ─────────────────────────────────────────────────────
    "consultations:read": ALL_ROLES,
    "consultations:create": MEDICAL_ROLES,
    "consultations:update": PRIVILEGED_ROLES,  # admin + doctor
    "consultations:delete": {ROLE_ADMIN},
    "consultations:ai_report": PRIVILEGED_ROLES,  # admin + doctor
    # ── Staff ─────────────────────────────────────────────────────────────
    "staff:read": MEDICAL_ROLES,
    "staff:create": {ROLE_ADMIN},
    "staff:update": {ROLE_ADMIN},
    "staff:delete": {ROLE_ADMIN},
    # ── Services ──────────────────────────────────────────────────────────
    "services:read": MEDICAL_ROLES,
    "services:create": {ROLE_ADMIN},
    "services:update": {ROLE_ADMIN},
    "services:delete": {ROLE_ADMIN},
    # ── AI Chat ───────────────────────────────────────────────────────────
    "ai:prompt_select": ALL_ROLES,  # Tous peuvent faire des SELECT via IA
    "ai:prompt_write": MEDICAL_ROLES,  # INSERT/UPDATE via IA
    "ai:prompt_delete": {ROLE_ADMIN},  # DELETE via IA
    "ai:prompt_ddl": {ROLE_ADMIN},  # DDL via IA
    "ai:execute_select": ALL_ROLES,
    "ai:execute_write": MEDICAL_ROLES,
    "ai:execute_delete": {ROLE_ADMIN},
    "ai:history_own": ALL_ROLES,  # Voir son propre historique
    "ai:history_all": {ROLE_ADMIN},  # Voir l'historique de tous
    # ── Métriques ─────────────────────────────────────────────────────────
    "metrics:read": PRIVILEGED_ROLES,
    "metrics:evaluation": {ROLE_ADMIN},
    "metrics:models": PRIVILEGED_ROLES,
    # ── Export ────────────────────────────────────────────────────────────
    "export:pdf": MEDICAL_ROLES,
    "export:csv": MEDICAL_ROLES,
    # ── Administration ────────────────────────────────────────────────────
    "admin:users": {ROLE_ADMIN},
    "admin:full": {ROLE_ADMIN},
}


# ─────────────────────────────────────────────────────────────────────────────
#  Fonctions utilitaires
# ─────────────────────────────────────────────────────────────────────────────


def can(role: str, permission: str) -> bool:
    """
    Vérifie si un rôle possède une permission donnée.

    Args:
        role:       Rôle de l'utilisateur ('admin', 'doctor', 'staff', 'patient').
        permission: Permission à vérifier (ex: 'patients:create').

    Returns:
        True si autorisé, False sinon.

    Usage:
        if can(user.role, 'consultations:update'):
            ...
    """
    allowed_roles = RESOURCE_PERMISSIONS.get(permission, set())
    return role in allowed_roles


def can_sql(role: str, action: str) -> bool:
    """
    Vérifie si un rôle peut exécuter une action SQL.

    Args:
        role:   Rôle de l'utilisateur.
        action: Action SQL ('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DDL').

    Returns:
        True si autorisé.
    """
    allowed_roles = SQL_PERMISSIONS.get(action.upper(), set())
    return role in allowed_roles


def check_sql_access(role: str, action: str) -> Optional[str]:
    """
    Vérifie l'accès SQL et retourne un message d'erreur si refusé.

    Args:
        role:   Rôle de l'utilisateur.
        action: Action SQL détectée.

    Returns:
        None si autorisé, message d'erreur sinon.
    """
    action_upper = action.upper()

    if action_upper == "SELECT":
        return None  # Tous les rôles peuvent lire

    if action_upper in ("INSERT", "UPDATE"):
        if role not in MEDICAL_ROLES:
            return (
                f"Accès refusé : les opérations {action_upper} nécessitent "
                f"le rôle doctor, staff ou admin. "
                f"Votre rôle '{role}' est limité à la lecture (SELECT)."
            )

    if action_upper == "DELETE":
        if role != ROLE_ADMIN:
            return (
                "Accès refusé : seuls les administrateurs peuvent supprimer des données. "
                f"Votre rôle '{role}' ne permet pas les opérations DELETE."
            )

    if action_upper in ("CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME", "DDL"):
        if role != ROLE_ADMIN:
            return (
                f"Accès refusé : les opérations DDL ({action_upper}) sont réservées "
                f"aux administrateurs. Votre rôle '{role}' ne permet pas de modifier "
                "la structure de la base de données."
            )

    return None


def get_patient_filter(user) -> Optional[int]:
    """
    Pour un patient, retourne son id_patient pour filtrer ses propres données.
    Pour les autres rôles, retourne None (pas de filtre).

    Args:
        user: Instance User courante.

    Returns:
        id_patient si rôle patient, None sinon.
    """
    if user.role == ROLE_PATIENT:
        return user.id_patient
    return None


def get_role_description(role: str) -> dict:
    """
    Retourne la description complète des permissions d'un rôle.

    Args:
        role: Rôle à décrire.

    Returns:
        Dictionnaire avec label, description et liste des permissions.
    """
    descriptions = {
        ROLE_ADMIN: {
            "label": "Administrateur",
            "description": "Accès complet à toutes les fonctionnalités, gestion des utilisateurs et DDL",
            "color": "red",
            "icon": "admin_panel_settings",
            "permissions": [
                p for p, roles in RESOURCE_PERMISSIONS.items() if role in roles
            ],
            "sql_actions": ["SELECT", "INSERT", "UPDATE", "DELETE", "DDL"],
        },
        ROLE_DOCTOR: {
            "label": "Médecin",
            "description": "Gestion des consultations, dossiers médicaux, rapports IA et requêtes SQL",
            "color": "blue",
            "icon": "medical_services",
            "permissions": [
                p for p, roles in RESOURCE_PERMISSIONS.items() if role in roles
            ],
            "sql_actions": ["SELECT", "INSERT", "UPDATE"],
        },
        ROLE_STAFF: {
            "label": "Staff Médical",
            "description": "Gestion des dossiers patients, création de consultations, requêtes de lecture/écriture",
            "color": "green",
            "icon": "badge",
            "permissions": [
                p for p, roles in RESOURCE_PERMISSIONS.items() if role in roles
            ],
            "sql_actions": ["SELECT", "INSERT", "UPDATE"],
        },
        ROLE_PATIENT: {
            "label": "Patient",
            "description": "Consultation de ses propres données médicales uniquement (lecture seule)",
            "color": "gray",
            "icon": "person",
            "permissions": [
                p for p, roles in RESOURCE_PERMISSIONS.items() if role in roles
            ],
            "sql_actions": ["SELECT"],
            "restriction": "Accès limité à ses propres données (id_patient filtré automatiquement)",
        },
    }
    return descriptions.get(role, {"label": role, "permissions": [], "sql_actions": []})
