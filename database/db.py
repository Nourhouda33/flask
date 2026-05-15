"""
Instance SQLAlchemy partagée + utilitaires base de données.
Importée par tous les modèles et par app.py pour l'initialisation.
"""

import logging
from typing import Optional
from flask_sqlalchemy import SQLAlchemy

logger = logging.getLogger(__name__)

# Instance unique — initialisée dans create_app() via db.init_app(app)
db = SQLAlchemy()


def get_db_schema() -> str:
    """
    Retourne le schéma de la base de données sous forme de texte structuré.
    Utilisé par le pipeline IA (Text2SQL) pour contextualiser les requêtes LLM.

    Le schéma décrit chaque table, ses colonnes (type, nullable, clé),
    et ses relations FK — sans données sensibles.

    Returns:
        Chaîne de caractères décrivant le schéma complet.
    """
    schema = """
=== SCHÉMA BASE DE DONNÉES : healthcare_ai_platform ===
Encodage : utf8mb4 | Moteur : InnoDB | MySQL 8.0+

────────────────────────────────────────────────────────
TABLE: Service
  Description: Services hospitaliers (Cardiologie, Neurologie, etc.)
  Colonnes:
    - id_service   INT          PK AUTO_INCREMENT
    - service_name VARCHAR(100) NOT NULL UNIQUE
  Relations:
    - Medical_staff.id_service → Service.id_service (FK, SET NULL)

────────────────────────────────────────────────────────
TABLE: Medical_staff
  Description: Personnel médical rattaché à un service
  Colonnes:
    - id_staff       INT          PK AUTO_INCREMENT
    - name_staff     VARCHAR(150) NOT NULL
    - position_staff ENUM('Doctor','Nurse','Technician','Administrator') NOT NULL DEFAULT 'Doctor'
    - speciality     VARCHAR(100) NULL
    - id_service     INT          NULL FK→Service.id_service
    - email          VARCHAR(150) NULL UNIQUE
    - phone          VARCHAR(20)  NULL
    - created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
  Relations:
    - Consultation.id_staff → Medical_staff.id_staff (FK, SET NULL)
    - Users.id_staff        → Medical_staff.id_staff (FK, SET NULL)

────────────────────────────────────────────────────────
TABLE: Patient
  Description: Données démographiques des patients
  Colonnes:
    - id_patient INT          PK AUTO_INCREMENT
    - first_name VARCHAR(100) NOT NULL
    - last_name  VARCHAR(100) NOT NULL
    - birthdate  DATE         NOT NULL
    - gender     ENUM('Male','Female') NULL
    - age        INT          VIRTUAL (TIMESTAMPDIFF(YEAR, birthdate, CURDATE()))
    - email      VARCHAR(150) NULL UNIQUE
    - phone      VARCHAR(20)  NULL
    - created_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
  Relations:
    - Medical_records.id_patient → Patient.id_patient (FK, CASCADE)
    - Consultation.id_patient    → Patient.id_patient (FK, SET NULL)
    - Users.id_patient           → Patient.id_patient (FK, SET NULL)

────────────────────────────────────────────────────────
TABLE: Medical_records
  Description: Dossier médical complet (1-to-1 avec Patient)
  Colonnes:
    - id_record        INT       PK AUTO_INCREMENT
    - id_patient       INT       NOT NULL UNIQUE FK→Patient.id_patient (CASCADE)
    - allergies        TEXT      NULL
    - chronic_diseases TEXT      NULL
    - blood_group      ENUM('A+','A-','B+','B-','AB+','AB-','O+','O-') NULL
    - medical_history  TEXT      NULL
    - last_updated     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP

────────────────────────────────────────────────────────
TABLE: Consultation
  Description: Consultations médicales (diagnostic, traitement, rapport)
  Colonnes:
    - id_consultation INT      PK AUTO_INCREMENT
    - diagnosis       TEXT     NOT NULL
    - treatment       TEXT     NULL
    - medical_report  TEXT     NULL
    - date            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    - id_staff        INT      NULL FK→Medical_staff.id_staff (SET NULL)
    - id_patient      INT      NULL FK→Patient.id_patient (SET NULL)

────────────────────────────────────────────────────────
TABLE: Users
  Description: Comptes utilisateurs avec rôles RBAC
  Colonnes:
    - id_user       INT          PK AUTO_INCREMENT
    - username      VARCHAR(100) NOT NULL UNIQUE
    - email         VARCHAR(150) NOT NULL UNIQUE
    - password_hash VARCHAR(255) NOT NULL
    - role          ENUM('admin','doctor','staff','patient') NOT NULL DEFAULT 'staff'
    - id_staff      INT          NULL FK→Medical_staff.id_staff (SET NULL)
    - id_patient    INT          NULL FK→Patient.id_patient (SET NULL)
    - is_active     BOOLEAN      NOT NULL DEFAULT TRUE
    - created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP

────────────────────────────────────────────────────────
TABLE: AI_Query_Logs
  Description: Historique des requêtes IA (Text2SQL) avec métriques
  Colonnes:
    - id_log           INT          PK AUTO_INCREMENT
    - user_id          INT          NULL FK→Users.id_user (SET NULL)
    - prompt           TEXT         NOT NULL
    - detected_intent  VARCHAR(50)  NULL
    - detected_tables  JSON         NULL
    - generated_sql    TEXT         NULL
    - execution_result TEXT         NULL
    - exact_match      BOOLEAN      NULL
    - confidence_score FLOAT        NULL
    - latency_ms       INT          NULL
    - created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP

────────────────────────────────────────────────────────
VUES DISPONIBLES:
  - v_patient_full        : Patient + Medical_records (LEFT JOIN)
  - v_consultation_full   : Consultation + Patient + Medical_staff + Service
  - v_dashboard_stats     : Statistiques globales agrégées

RÈGLES IMPORTANTES POUR LA GÉNÉRATION SQL:
  1. Utiliser les noms de tables exacts (casse respectée) : Service, Medical_staff, Patient, etc.
  2. La colonne `age` dans Patient est VIRTUELLE — ne pas l'utiliser dans INSERT/UPDATE.
  3. Pour les noms complets des patients : CONCAT(first_name, ' ', last_name).
  4. Pour filtrer par médecin : JOIN Medical_staff ON id_staff.
  5. Les dates de consultation sont dans la colonne `date` (DATETIME).
  6. Toujours utiliser utf8mb4 pour les comparaisons de chaînes.
"""
    return schema.strip()


def check_db_connection() -> dict:
    """
    Vérifie la connexion à la base de données.
    Utile pour le health check de l'application.

    Returns:
        Dictionnaire {"connected": bool, "error": str|None}.
    """
    try:
        db.session.execute(db.text("SELECT 1"))
        return {"connected": True, "error": None}
    except Exception as exc:
        logger.error("Erreur de connexion DB : %s", str(exc))
        return {"connected": False, "error": str(exc)}


def paginate_query(query, page: int = 1, per_page: int = 20, max_per_page: int = 100):
    """
    Pagine une requête SQLAlchemy de manière uniforme.

    Args:
        query:       Requête SQLAlchemy à paginer.
        page:        Numéro de page (1-indexé).
        per_page:    Nombre d'éléments par page.
        max_per_page: Limite maximale pour per_page.

    Returns:
        Objet Pagination SQLAlchemy avec les métadonnées.
    """
    page     = max(1, page)
    per_page = min(max(1, per_page), max_per_page)
    return query.paginate(page=page, per_page=per_page, error_out=False)


def pagination_meta(pagination) -> dict:
    """
    Extrait les métadonnées de pagination en dictionnaire JSON-safe.

    Args:
        pagination: Objet Pagination SQLAlchemy.

    Returns:
        Dictionnaire avec page, per_page, total, pages, has_next, has_prev.
    """
    return {
        "page":     pagination.page,
        "per_page": pagination.per_page,
        "total":    pagination.total,
        "pages":    pagination.pages,
        "has_next": pagination.has_next,
        "has_prev": pagination.has_prev,
    }
