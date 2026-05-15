"""
Schema Builder — Healthcare AI Platform
Construit la représentation du schéma DB pour le pipeline IA.
Utilisé par le Table Matcher et le générateur SQL (Qwen) pour fournir
le contexte de la base de données sous forme structurée.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Schéma statique de la base healthcare_ai_platform
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA: Dict[str, Dict[str, Any]] = {
    "Service": {
        "description": "Services hospitaliers (Cardiologie, Neurologie, Oncologie, etc.)",
        "columns": {
            "id_service":   {"type": "INT",          "pk": True,  "nullable": False},
            "service_name": {"type": "VARCHAR(100)", "pk": False, "nullable": False, "unique": True},
        },
        "foreign_keys": [],
        "indexes": ["service_name"],
        "semantic_tags": ["service", "department", "hospital unit", "ward", "cardiologie",
                          "neurologie", "oncologie", "pédiatrie", "urgences"],
    },

    "Medical_staff": {
        "description": "Personnel médical : médecins, infirmiers, techniciens, administrateurs",
        "columns": {
            "id_staff":       {"type": "INT",          "pk": True,  "nullable": False},
            "name_staff":     {"type": "VARCHAR(150)", "pk": False, "nullable": False},
            "position_staff": {"type": "ENUM",         "pk": False, "nullable": False,
                               "values": ["Doctor", "Nurse", "Technician", "Administrator"]},
            "speciality":     {"type": "VARCHAR(100)", "pk": False, "nullable": True},
            "id_service":     {"type": "INT",          "pk": False, "nullable": True,
                               "fk": "Service.id_service"},
            "email":          {"type": "VARCHAR(150)", "pk": False, "nullable": True, "unique": True},
            "phone":          {"type": "VARCHAR(20)",  "pk": False, "nullable": True},
            "created_at":     {"type": "TIMESTAMP",    "pk": False, "nullable": False},
        },
        "foreign_keys": [
            {"column": "id_service", "references": "Service.id_service", "on_delete": "SET NULL"}
        ],
        "indexes": ["id_service", "position_staff", "name_staff"],
        "semantic_tags": ["doctor", "nurse", "staff", "physician", "medical personnel",
                          "médecin", "infirmier", "technicien", "spécialiste", "clinicien"],
    },

    "Patient": {
        "description": "Données démographiques des patients hospitalisés",
        "columns": {
            "id_patient": {"type": "INT",          "pk": True,  "nullable": False},
            "first_name": {"type": "VARCHAR(100)", "pk": False, "nullable": False},
            "last_name":  {"type": "VARCHAR(100)", "pk": False, "nullable": False},
            "birthdate":  {"type": "DATE",         "pk": False, "nullable": False},
            "gender":     {"type": "ENUM",         "pk": False, "nullable": True,
                           "values": ["Male", "Female"]},
            "age":        {"type": "INT VIRTUAL",  "pk": False, "nullable": True,
                           "computed": "TIMESTAMPDIFF(YEAR, birthdate, CURDATE())"},
            "email":      {"type": "VARCHAR(150)", "pk": False, "nullable": True, "unique": True},
            "phone":      {"type": "VARCHAR(20)",  "pk": False, "nullable": True},
            "created_at": {"type": "TIMESTAMP",    "pk": False, "nullable": False},
        },
        "foreign_keys": [],
        "indexes": ["last_name", "first_name", "birthdate", "gender"],
        "semantic_tags": ["patient", "person", "individual", "sick person", "malade",
                          "hospitalisé", "demographics", "âge", "genre"],
    },

    "Medical_records": {
        "description": "Dossiers médicaux : allergies, maladies chroniques, groupe sanguin, historique",
        "columns": {
            "id_record":        {"type": "INT",       "pk": True,  "nullable": False},
            "id_patient":       {"type": "INT",       "pk": False, "nullable": False,
                                 "fk": "Patient.id_patient"},
            "allergies":        {"type": "TEXT",      "pk": False, "nullable": True},
            "chronic_diseases": {"type": "TEXT",      "pk": False, "nullable": True},
            "blood_group":      {"type": "ENUM",      "pk": False, "nullable": True,
                                 "values": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]},
            "medical_history":  {"type": "TEXT",      "pk": False, "nullable": True},
            "last_updated":     {"type": "TIMESTAMP", "pk": False, "nullable": False},
        },
        "foreign_keys": [
            {"column": "id_patient", "references": "Patient.id_patient", "on_delete": "CASCADE"}
        ],
        "indexes": ["id_patient", "blood_group"],
        "semantic_tags": ["medical record", "dossier médical", "allergy", "allergie",
                          "chronic disease", "maladie chronique", "blood group", "groupe sanguin",
                          "history", "antécédents", "diabète", "hypertension", "asthme"],
    },

    "Consultation": {
        "description": "Consultations médicales : diagnostic, traitement, rapport, date",
        "columns": {
            "id_consultation": {"type": "INT",      "pk": True,  "nullable": False},
            "diagnosis":       {"type": "TEXT",     "pk": False, "nullable": False},
            "treatment":       {"type": "TEXT",     "pk": False, "nullable": True},
            "medical_report":  {"type": "TEXT",     "pk": False, "nullable": True},
            "date":            {"type": "DATETIME", "pk": False, "nullable": False},
            "id_staff":        {"type": "INT",      "pk": False, "nullable": True,
                                "fk": "Medical_staff.id_staff"},
            "id_patient":      {"type": "INT",      "pk": False, "nullable": True,
                                "fk": "Patient.id_patient"},
        },
        "foreign_keys": [
            {"column": "id_staff",   "references": "Medical_staff.id_staff",  "on_delete": "SET NULL"},
            {"column": "id_patient", "references": "Patient.id_patient",       "on_delete": "SET NULL"},
        ],
        "indexes": ["id_patient", "id_staff", "date"],
        "semantic_tags": ["consultation", "visit", "appointment", "diagnosis", "diagnostic",
                          "treatment", "traitement", "report", "rapport", "ordonnance"],
    },

    "Users": {
        "description": "Comptes utilisateurs avec rôles RBAC (admin/doctor/staff/patient)",
        "columns": {
            "id_user":       {"type": "INT",          "pk": True,  "nullable": False},
            "username":      {"type": "VARCHAR(100)", "pk": False, "nullable": False, "unique": True},
            "email":         {"type": "VARCHAR(150)", "pk": False, "nullable": False, "unique": True},
            "password_hash": {"type": "VARCHAR(255)", "pk": False, "nullable": False},
            "role":          {"type": "ENUM",         "pk": False, "nullable": False,
                              "values": ["admin", "doctor", "staff", "patient"]},
            "id_staff":      {"type": "INT",          "pk": False, "nullable": True,
                              "fk": "Medical_staff.id_staff"},
            "id_patient":    {"type": "INT",          "pk": False, "nullable": True,
                              "fk": "Patient.id_patient"},
            "is_active":     {"type": "BOOLEAN",      "pk": False, "nullable": False},
            "created_at":    {"type": "TIMESTAMP",    "pk": False, "nullable": False},
        },
        "foreign_keys": [
            {"column": "id_staff",   "references": "Medical_staff.id_staff", "on_delete": "SET NULL"},
            {"column": "id_patient", "references": "Patient.id_patient",     "on_delete": "SET NULL"},
        ],
        "indexes": ["role", "id_staff", "id_patient", "is_active"],
        "semantic_tags": ["user", "account", "utilisateur", "login", "role", "permission",
                          "authentication", "access control"],
    },

    "AI_Query_Logs": {
        "description": "Logs des requêtes IA : prompts, SQL généré, métriques d'évaluation",
        "columns": {
            "id_log":           {"type": "INT",         "pk": True,  "nullable": False},
            "user_id":          {"type": "INT",         "pk": False, "nullable": True,
                                 "fk": "Users.id_user"},
            "prompt":           {"type": "TEXT",        "pk": False, "nullable": False},
            "detected_intent":  {"type": "VARCHAR(50)", "pk": False, "nullable": True},
            "detected_tables":  {"type": "JSON",        "pk": False, "nullable": True},
            "generated_sql":    {"type": "TEXT",        "pk": False, "nullable": True},
            "execution_result": {"type": "TEXT",        "pk": False, "nullable": True},
            "exact_match":      {"type": "BOOLEAN",     "pk": False, "nullable": True},
            "confidence_score": {"type": "FLOAT",       "pk": False, "nullable": True},
            "latency_ms":       {"type": "INT",         "pk": False, "nullable": True},
            "created_at":       {"type": "TIMESTAMP",   "pk": False, "nullable": False},
        },
        "foreign_keys": [
            {"column": "user_id", "references": "Users.id_user", "on_delete": "SET NULL"}
        ],
        "indexes": ["user_id", "detected_intent", "created_at", "exact_match"],
        "semantic_tags": ["log", "query", "ai", "sql", "metrics", "evaluation",
                          "performance", "historique requêtes"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Relations entre tables (pour la construction des JOINs)
# ─────────────────────────────────────────────────────────────────────────────

RELATIONSHIPS: List[Dict[str, str]] = [
    {"from": "Medical_staff",   "to": "Service",       "join": "Medical_staff.id_service = Service.id_service"},
    {"from": "Medical_records", "to": "Patient",       "join": "Medical_records.id_patient = Patient.id_patient"},
    {"from": "Consultation",    "to": "Patient",       "join": "Consultation.id_patient = Patient.id_patient"},
    {"from": "Consultation",    "to": "Medical_staff", "join": "Consultation.id_staff = Medical_staff.id_staff"},
    {"from": "Consultation",    "to": "Service",       "join": "Medical_staff.id_service = Service.id_service"},
    {"from": "Users",           "to": "Medical_staff", "join": "Users.id_staff = Medical_staff.id_staff"},
    {"from": "Users",           "to": "Patient",       "join": "Users.id_patient = Patient.id_patient"},
    {"from": "AI_Query_Logs",   "to": "Users",         "join": "AI_Query_Logs.user_id = Users.id_user"},
]


# ─────────────────────────────────────────────────────────────────────────────
#  SchemaBuilder
# ─────────────────────────────────────────────────────────────────────────────

class SchemaBuilder:
    """
    Construit et expose le schéma de la base de données pour le pipeline IA.
    Fournit des méthodes pour le Table Matcher, le générateur SQL et l'API.
    """

    # ── Accès au schéma ───────────────────────────────────────────────────────

    def get_all_tables(self) -> List[str]:
        """Retourne la liste de toutes les tables."""
        return list(SCHEMA.keys())

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Retourne le schéma complet d'une table.

        Args:
            table_name: Nom de la table.

        Returns:
            Dictionnaire avec colonnes, FK, index, tags sémantiques.

        Raises:
            KeyError: Si la table n'existe pas.
        """
        if table_name not in SCHEMA:
            raise KeyError(f"Table '{table_name}' introuvable dans le schéma.")
        return SCHEMA[table_name]

    def get_columns(self, table_name: str) -> List[str]:
        """Retourne la liste des colonnes d'une table."""
        return list(SCHEMA[table_name]["columns"].keys())

    def get_column_info(self, table_name: str, column_name: str) -> Optional[Dict]:
        """Retourne les informations d'une colonne spécifique."""
        table = SCHEMA.get(table_name, {})
        return table.get("columns", {}).get(column_name)

    def get_semantic_tags(self, table_name: str) -> List[str]:
        """Retourne les tags sémantiques d'une table (pour le matching IA)."""
        return SCHEMA[table_name].get("semantic_tags", [])

    def get_all_semantic_tags(self) -> Dict[str, List[str]]:
        """Retourne tous les tags sémantiques indexés par table."""
        return {t: SCHEMA[t].get("semantic_tags", []) for t in SCHEMA}

    # ── Descriptions sémantiques ──────────────────────────────────────────────

    def get_full_schema(self) -> Dict[str, Any]:
        """
        Retourne le schéma complet avec toutes les tables, colonnes, types et FK.

        Returns:
            Dictionnaire {table: {columns, foreign_keys, indexes, description}}.
        """
        return {
            "tables":        SCHEMA,
            "relationships": RELATIONSHIPS,
            "table_count":   len(SCHEMA),
            "total_columns": sum(len(t["columns"]) for t in SCHEMA.values()),
        }

    def get_table_descriptions(self) -> Dict[str, str]:
        """
        Retourne les descriptions sémantiques de toutes les tables.
        Utilisé pour améliorer le matching FAISS.

        Returns:
            Dictionnaire {table_name: description_en_langage_naturel}.
        """
        return {
            "Service": (
                "Service hospitalier ou département médical. "
                "Contient les unités de soins : Cardiologie, Neurologie, Oncologie, Pédiatrie, Urgences. "
                "Chaque service regroupe du personnel médical spécialisé."
            ),
            "Medical_staff": (
                "Personnel médical de l'hôpital : médecins (doctors), infirmiers (nurses), "
                "techniciens et administrateurs. Contient le nom complet, la spécialité médicale "
                "(cardiologie, neurologie, chirurgie...), le poste et le service d'appartenance."
            ),
            "Patient": (
                "Informations personnelles et démographiques des patients hospitalisés. "
                "Contient le prénom, nom de famille, date de naissance, âge calculé automatiquement, "
                "genre (Male/Female), email et téléphone de contact."
            ),
            "Medical_records": (
                "Dossier médical complet d'un patient (relation 1-to-1 avec Patient). "
                "Contient les allergies connues, les maladies chroniques (diabète, hypertension, asthme, "
                "cancer, BPCO, insuffisance rénale), le groupe sanguin (A+/A-/B+/B-/AB+/AB-/O+/O-) "
                "et l'historique médical complet (antécédents, chirurgies, hospitalisations)."
            ),
            "Consultation": (
                "Consultation médicale entre un patient et un médecin. "
                "Contient le diagnostic établi, le traitement prescrit (médicaments, thérapies), "
                "le rapport médical détaillé et la date/heure de la consultation. "
                "Liée à un patient et à un membre du personnel médical."
            ),
            "Users": (
                "Comptes utilisateurs du système avec rôles d'accès RBAC. "
                "Rôles disponibles : admin (administrateur), doctor (médecin), "
                "staff (personnel), patient. Gère l'authentification et les permissions."
            ),
            "AI_Query_Logs": (
                "Historique des requêtes IA Text2SQL soumises au système. "
                "Contient le prompt utilisateur, l'intention détectée, le SQL généré, "
                "le résultat d'exécution et les métriques de performance "
                "(score de confiance, latence, exact match)."
            ),
        }

    # ── Jointures ─────────────────────────────────────────────────────────────

    def get_join_path(self, table_a: str, table_b: str) -> Optional[str]:
        """
        Retourne la condition JOIN directe entre deux tables si elle existe.

        Args:
            table_a: Première table.
            table_b: Deuxième table.

        Returns:
            Condition JOIN SQL ou None.
        """
        for rel in RELATIONSHIPS:
            if (rel["from"] == table_a and rel["to"] == table_b) or \
               (rel["from"] == table_b and rel["to"] == table_a):
                return rel["join"]
        return None

    def get_join_paths(self, table_a: str, table_b: str) -> List[str]:
        """
        Retourne tous les chemins de jointure entre deux tables
        (directs et indirects via table intermédiaire).

        Args:
            table_a: Première table.
            table_b: Deuxième table.

        Returns:
            Liste de conditions JOIN SQL.
        """
        paths = []

        # Jointure directe
        direct = self.get_join_path(table_a, table_b)
        if direct:
            paths.append(direct)

        # Jointures indirectes (via une table intermédiaire)
        for intermediate in SCHEMA:
            if intermediate in (table_a, table_b):
                continue
            path_a = self.get_join_path(table_a, intermediate)
            path_b = self.get_join_path(intermediate, table_b)
            if path_a and path_b:
                paths.append(f"-- Via {intermediate}: {path_a} AND {path_b}")

        return paths

    def get_all_join_paths(self, tables: List[str]) -> List[str]:
        """
        Retourne tous les chemins de jointure entre une liste de tables.

        Args:
            tables: Liste de tables à connecter.

        Returns:
            Liste de conditions JOIN SQL.
        """
        paths = []
        for i, table_a in enumerate(tables):
            for table_b in tables[i + 1:]:
                join = self.get_join_path(table_a, table_b)
                if join:
                    paths.append(join)
        return paths

    # ── Contexte SQL ──────────────────────────────────────────────────────────

    def build_schema_context(self, tables: List[str]) -> str:
        """
        Construit un contexte textuel du schéma pour le prompt LLM.

        Args:
            tables: Liste des tables à inclure.

        Returns:
            Chaîne formatée décrivant le schéma pour le LLM.
        """
        lines = ["-- Database: healthcare_ai_platform", ""]
        for table in tables:
            if table not in SCHEMA:
                continue
            schema = SCHEMA[table]
            lines.append(f"-- {schema['description']}")
            lines.append(f"CREATE TABLE `{table}` (")
            col_lines = []
            for col_name, col_info in schema["columns"].items():
                col_type = col_info["type"].replace(" VIRTUAL", "")
                pk_str   = " PRIMARY KEY" if col_info.get("pk") else ""
                fk_str   = f" -- FK → {col_info['fk']}" if col_info.get("fk") else ""
                virtual  = " -- VIRTUAL" if "VIRTUAL" in col_info["type"] else ""
                col_lines.append(f"    `{col_name}` {col_type}{pk_str}{fk_str}{virtual}")
            lines.append(",\n".join(col_lines))
            lines.append(");\n")

        # Ajouter les relations pertinentes
        relevant_rels = [
            r for r in RELATIONSHIPS
            if r["from"] in tables and r["to"] in tables
        ]
        if relevant_rels:
            lines.append("-- Jointures disponibles :")
            for rel in relevant_rels:
                lines.append(f"-- JOIN {rel['from']} ON {rel['join']}")

        return "\n".join(lines)

    def get_full_schema_dict(self) -> Dict[str, Any]:
        """Retourne le schéma complet sous forme de dictionnaire (pour l'API)."""
        return {
            "tables":        SCHEMA,
            "relationships": RELATIONSHIPS,
        }

    # ── Validation ────────────────────────────────────────────────────────────

    def is_valid_table(self, table_name: str) -> bool:
        """Vérifie si une table existe dans le schéma."""
        return table_name in SCHEMA

    def is_valid_column(self, table_name: str, column_name: str) -> bool:
        """Vérifie si une colonne existe dans une table."""
        return (
            table_name in SCHEMA
            and column_name in SCHEMA[table_name].get("columns", {})
        )

    def is_virtual_column(self, table_name: str, column_name: str) -> bool:
        """Vérifie si une colonne est virtuelle (GENERATED ALWAYS)."""
        col_info = self.get_column_info(table_name, column_name)
        if col_info is None:
            return False
        return "VIRTUAL" in col_info.get("type", "") or col_info.get("computed") is not None


# ─────────────────────────────────────────────────────────────────────────────
#  Instance singleton
# ─────────────────────────────────────────────────────────────────────────────

schema_builder = SchemaBuilder()
