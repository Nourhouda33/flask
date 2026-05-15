"""
SQL Generator — Healthcare AI Platform
Génère des requêtes SQL à partir de langage naturel via Qwen2.5-Coder-7B-Instruct.

Architecture :
  SQLGenerator  → classe principale avec few-shot prompting
  _build_prompt → construit le prompt optimisé pour Text2SQL médical
  _extract_sql  → extrait le SQL propre depuis la réponse du modèle
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Few-shot examples — 5 paires (question médicale → SQL attendu)
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    # 1. SELECT simple avec filtre
    {
        "question": "Liste tous les patients de sexe féminin",
        "sql": (
            "SELECT id_patient, first_name, last_name, birthdate, age, email, phone\n"
            "FROM Patient\n"
            "WHERE gender = 'Female'\n"
            "ORDER BY last_name, first_name;"
        ),
    },
    # 2. SELECT avec JOIN et filtre sur maladie chronique
    {
        "question": "Affiche les patients atteints de diabète avec leur groupe sanguin",
        "sql": (
            "SELECT p.id_patient, p.first_name, p.last_name, p.age,\n"
            "       mr.blood_group, mr.chronic_diseases\n"
            "FROM Patient p\n"
            "INNER JOIN Medical_records mr ON mr.id_patient = p.id_patient\n"
            "WHERE mr.chronic_diseases LIKE '%diabète%'\n"
            "ORDER BY p.last_name;"
        ),
    },
    # 3. SELECT avec GROUP BY et COUNT
    {
        "question": "Combien de consultations par médecin ce mois-ci ?",
        "sql": (
            "SELECT ms.id_staff, ms.name_staff, ms.speciality,\n"
            "       COUNT(c.id_consultation) AS nb_consultations\n"
            "FROM Medical_staff ms\n"
            "LEFT JOIN Consultation c ON c.id_staff = ms.id_staff\n"
            "    AND MONTH(c.date) = MONTH(CURDATE())\n"
            "    AND YEAR(c.date)  = YEAR(CURDATE())\n"
            "WHERE ms.position_staff = 'Doctor'\n"
            "GROUP BY ms.id_staff, ms.name_staff, ms.speciality\n"
            "ORDER BY nb_consultations DESC;"
        ),
    },
    # 4. INSERT patient
    {
        "question": "Ajouter un nouveau patient : Marie Dupont, née le 1990-05-15, femme",
        "sql": (
            "INSERT INTO Patient (first_name, last_name, birthdate, gender)\n"
            "VALUES ('Marie', 'Dupont', '1990-05-15', 'Female');"
        ),
    },
    # 5. SELECT complexe avec sous-requête
    {
        "question": "Quels patients n'ont aucune consultation enregistrée ?",
        "sql": (
            "SELECT p.id_patient, p.first_name, p.last_name, p.age, p.gender\n"
            "FROM Patient p\n"
            "WHERE p.id_patient NOT IN (\n"
            "    SELECT DISTINCT id_patient\n"
            "    FROM Consultation\n"
            "    WHERE id_patient IS NOT NULL\n"
            ")\n"
            "ORDER BY p.last_name, p.first_name;"
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
#  Prompt système Qwen2.5-Coder Text2SQL
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_SQL_GENERATOR = """Tu es un expert SQL spécialisé dans les bases de données médicales hospitalières.
Tu génères des requêtes MySQL 8.0 précises, optimisées et sécurisées.

RÈGLES ABSOLUES:
1. Retourner UNIQUEMENT la requête SQL, sans explication ni markdown
2. Ne jamais utiliser la colonne `age` dans INSERT ou UPDATE (colonne VIRTUELLE)
3. Respecter la casse exacte des noms de tables : Patient, Medical_staff, Medical_records, Consultation, Service
4. Utiliser des backticks pour les noms de colonnes avec caractères spéciaux
5. Toujours terminer par un point-virgule
6. Pour les recherches textuelles : utiliser LIKE avec %valeur%
7. Pour les noms complets : CONCAT(first_name, ' ', last_name)
8. Utiliser LEFT JOIN quand des données peuvent être NULL
9. Ajouter ORDER BY pour les SELECT (lisibilité)
10. Limiter les résultats avec LIMIT si la requête peut retourner beaucoup de lignes

TYPES DE REQUÊTES SUPPORTÉES:
- SELECT simple, avec WHERE, JOIN, GROUP BY, HAVING, ORDER BY, LIMIT
- SELECT avec sous-requêtes (IN, EXISTS, NOT IN)
- SELECT avec fonctions d'agrégation (COUNT, AVG, MAX, MIN, SUM)
- INSERT avec valeurs explicites
- UPDATE avec WHERE précis
- DELETE avec WHERE précis (jamais sans WHERE)

JOINTURES DISPONIBLES:
- Patient ↔ Medical_records : Medical_records.id_patient = Patient.id_patient
- Patient ↔ Consultation    : Consultation.id_patient = Patient.id_patient
- Medical_staff ↔ Consultation : Consultation.id_staff = Medical_staff.id_staff
- Medical_staff ↔ Service   : Medical_staff.id_service = Service.id_service"""


# ─────────────────────────────────────────────────────────────────────────────
#  SQLGenerator
# ─────────────────────────────────────────────────────────────────────────────

class SQLGenerator:
    """
    Génère des requêtes SQL à partir de langage naturel via Qwen2.5-Coder.

    Usage:
        generator = SQLGenerator(ollama_client, model="qwen2.5-coder:7b-instruct")
        sql = generator.generate(
            prompt="Liste les patients diabétiques",
            schema_context=schema_str,
            intent_info=intent_dict,
        )
    """

    def __init__(
        self,
        ollama_client,
        model:       str   = "qwen2.5-coder:7b-instruct",
        temperature: float = 0.05,
        max_tokens:  int   = 512,
    ):
        """
        Args:
            ollama_client: Instance OllamaClient.
            model:         Modèle Qwen à utiliser.
            temperature:   Température (très basse pour SQL déterministe).
            max_tokens:    Nombre maximum de tokens générés.
        """
        self.client      = ollama_client
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def generate(
        self,
        prompt:         str,
        schema_context: str,
        intent_info:    dict,
    ) -> str:
        """
        Génère une requête SQL à partir d'une requête en langage naturel.

        Args:
            prompt:         Requête utilisateur en langage naturel.
            schema_context: Contexte du schéma DB (tables, colonnes, FK).
            intent_info:    Résultat de l'IntentAgent (action, tables, filters...).

        Returns:
            Requête SQL propre (sans markdown ni commentaires superflus).
        """
        user_prompt = self._build_prompt(prompt, schema_context, intent_info)

        try:
            raw_response = self.client.generate(
                model=self.model,
                prompt=user_prompt,
                system=SYSTEM_SQL_GENERATOR,
                temperature=self.temperature,
            )
            sql = self._extract_sql(raw_response)
            logger.info(
                "SQL généré — action=%s tables=%s longueur=%d chars",
                intent_info.get("action"), intent_info.get("tables"), len(sql),
            )
            return sql

        except Exception as exc:
            logger.error("Erreur SQLGenerator : %s", str(exc), exc_info=True)
            raise

    def _build_prompt(
        self,
        prompt:         str,
        schema_context: str,
        intent_info:    dict,
    ) -> str:
        """
        Construit le prompt complet pour Qwen avec few-shot examples.

        Structure :
          1. Schéma DB (contexte)
          2. Few-shot examples (5 paires question → SQL)
          3. Informations d'intention (action, tables, filtres détectés)
          4. Question de l'utilisateur
        """
        action     = intent_info.get("action", "SELECT")
        tables     = intent_info.get("tables", [])
        filters    = intent_info.get("filters", [])
        attributes = intent_info.get("attributes", [])

        # ── Section schéma ─────────────────────────────────────────────────
        lines = [
            "### SCHÉMA DE LA BASE DE DONNÉES",
            schema_context,
            "",
            "### EXEMPLES FEW-SHOT",
        ]

        # ── Few-shot examples ──────────────────────────────────────────────
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            lines.append(f"-- Exemple {i}")
            lines.append(f"-- Question: {ex['question']}")
            lines.append(ex["sql"])
            lines.append("")

        # ── Contexte d'intention ───────────────────────────────────────────
        lines.append("### CONTEXTE DE LA REQUÊTE ACTUELLE")
        lines.append(f"-- Action détectée : {action}")
        if tables:
            lines.append(f"-- Tables concernées : {', '.join(tables)}")
        if attributes:
            lines.append(f"-- Attributs mentionnés : {', '.join(attributes)}")
        if filters:
            for f in filters:
                lines.append(
                    f"-- Filtre détecté : {f.get('column')} {f.get('operator')} {f.get('value')}"
                )

        # ── Instructions spécifiques selon l'action ────────────────────────
        if action == "SELECT":
            lines.append("-- Générer un SELECT avec les colonnes pertinentes et les JOINs nécessaires")
        elif action == "INSERT":
            lines.append("-- Générer un INSERT avec les colonnes NOT NULL obligatoires")
            lines.append("-- NE PAS inclure la colonne `age` (VIRTUELLE)")
        elif action == "UPDATE":
            lines.append("-- Générer un UPDATE avec une clause WHERE précise")
        elif action == "DELETE":
            lines.append("-- Générer un DELETE avec une clause WHERE précise (jamais sans WHERE)")

        lines.append("")
        lines.append("### QUESTION")
        lines.append(f"-- {prompt}")
        lines.append("")
        lines.append("### SQL (retourner UNIQUEMENT la requête SQL, sans explication)")

        return "\n".join(lines)

    @staticmethod
    def _extract_sql(raw_response: str) -> str:
        """
        Extrait la requête SQL propre depuis la réponse brute du modèle.
        Gère les cas où le modèle ajoute du markdown ou du texte superflu.

        Args:
            raw_response: Réponse brute de Qwen.

        Returns:
            Requête SQL nettoyée.
        """
        if not raw_response:
            return ""

        text = raw_response.strip()

        # Stratégie 1 : extraire depuis un bloc ```sql ... ```
        sql_block = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if sql_block:
            return sql_block.group(1).strip()

        # Stratégie 2 : extraire depuis un bloc ``` ... ```
        generic_block = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if generic_block:
            candidate = generic_block.group(1).strip()
            if _looks_like_sql(candidate):
                return candidate

        # Stratégie 3 : chercher la première ligne SQL
        lines = text.split("\n")
        sql_lines = []
        in_sql = False

        for line in lines:
            stripped = line.strip()
            # Détecter le début d'une requête SQL
            if not in_sql and _starts_sql_statement(stripped):
                in_sql = True
            if in_sql:
                # Arrêter si on rencontre du texte non-SQL
                if stripped and not _is_sql_line(stripped):
                    break
                sql_lines.append(line)

        if sql_lines:
            return "\n".join(sql_lines).strip()

        # Stratégie 4 : retourner le texte brut nettoyé
        # Supprimer les lignes de commentaire non-SQL
        clean_lines = [
            line for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
        return "\n".join(clean_lines).strip()


def _starts_sql_statement(line: str) -> bool:
    """Vérifie si une ligne commence une instruction SQL."""
    upper = line.upper().lstrip()
    return any(upper.startswith(kw) for kw in (
        "SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "DROP", "ALTER"
    ))


def _is_sql_line(line: str) -> bool:
    """Vérifie si une ligne ressemble à du SQL (pas du texte explicatif)."""
    stripped = line.strip()
    if not stripped:
        return True  # Ligne vide OK
    if stripped.startswith("--"):
        return True  # Commentaire SQL OK
    # Rejeter les lignes qui ressemblent à du texte naturel
    natural_patterns = [
        r"^(voici|here|this|la requête|the query|explication|note:)",
        r"^[A-Z][a-z].*:$",  # "Explication :"
    ]
    for pattern in natural_patterns:
        if re.match(pattern, stripped, re.IGNORECASE):
            return False
    return True


def _looks_like_sql(text: str) -> bool:
    """Vérifie si un texte ressemble à du SQL."""
    upper = text.upper()
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "JOIN"]
    return any(kw in upper for kw in sql_keywords)
