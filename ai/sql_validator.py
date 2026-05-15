"""
SQL Validator — Healthcare AI Platform
Valide et corrige automatiquement les requêtes SQL générées par Qwen/SQLCoder.

Architecture :
  SQLValidator      → classe principale (validation + correction)
  _StaticValidator  → vérifications statiques (tables, colonnes, syntaxe)
  _LLMFixer         → correction via SQLCoder si disponible
"""

import re
import logging
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Tables valides du schéma
VALID_TABLES = {
    "Patient", "Medical_records", "Consultation",
    "Medical_staff", "Service", "Users", "AI_Query_Logs",
}

# Colonnes virtuelles — interdites dans INSERT/UPDATE
VIRTUAL_COLUMNS = {
    ("Patient", "age"),
}

# Colonnes NOT NULL par table (obligatoires dans INSERT)
NOT_NULL_COLUMNS: Dict[str, List[str]] = {
    "Patient":         ["first_name", "last_name", "birthdate"],
    "Medical_records": ["id_patient"],
    "Consultation":    ["diagnosis", "date"],
    "Medical_staff":   ["name_staff", "position_staff"],
    "Service":         ["service_name"],
    "Users":           ["username", "email", "password_hash", "role"],
}

# Corrections automatiques courantes (regex → remplacement)
AUTO_CORRECTIONS = [
    # Supprimer les backticks inutiles autour des mots-clés SQL
    (r"`(SELECT|FROM|WHERE|JOIN|ON|AND|OR|GROUP BY|ORDER BY|HAVING|LIMIT)`", r"\1"),
    # Corriger la casse des noms de tables connus
    (r"\bpatient\b(?!s)", "Patient"),
    (r"\bmedical_staff\b", "Medical_staff"),
    (r"\bmedical_records\b", "Medical_records"),
    (r"\bconsultation\b(?!s)", "Consultation"),
    (r"\bservice\b(?! )", "Service"),
    # Supprimer les points-virgules multiples
    (r";+\s*$", ";"),
    # Normaliser les espaces multiples
    (r"  +", " "),
]

# Prompt système SQLCoder pour la correction
SYSTEM_SQLCODER_FIX = """Tu es un expert SQL MySQL 8.0. Corrige la requête SQL fournie.
Retourne UNIQUEMENT la requête SQL corrigée, sans explication ni markdown.
Règles :
- Respecter la casse des tables : Patient, Medical_staff, Medical_records, Consultation, Service
- Ne jamais utiliser la colonne `age` dans INSERT/UPDATE (VIRTUELLE)
- Toujours terminer par un point-virgule
- Utiliser des backticks pour les noms de colonnes réservés"""


# ─────────────────────────────────────────────────────────────────────────────
#  SQLValidator
# ─────────────────────────────────────────────────────────────────────────────

class SQLValidator:
    """
    Valide et corrige automatiquement les requêtes SQL.

    Étapes de validation :
      1. Vérification syntaxique basique (mots-clés, structure)
      2. Vérification des tables (existent dans le schéma)
      3. Vérification des colonnes virtuelles (pas dans INSERT/UPDATE)
      4. Corrections automatiques par regex
      5. Correction via SQLCoder si disponible et si erreurs persistantes

    Usage:
        validator = SQLValidator(ollama_client, sqlcoder_model="sqlcoder")
        result = validator.validate_and_fix(sql, schema)
        if result["valid"]:
            execute(result["fixed_sql"])
    """

    def __init__(
        self,
        ollama_client=None,
        sqlcoder_model: str = "sqlcoder",
        use_llm_fix:    bool = True,
    ):
        """
        Args:
            ollama_client:  Instance OllamaClient (optionnel, pour la correction LLM).
            sqlcoder_model: Modèle SQLCoder pour la correction.
            use_llm_fix:    Utiliser SQLCoder pour corriger les erreurs non réparables.
        """
        self.client         = ollama_client
        self.sqlcoder_model = sqlcoder_model
        self.use_llm_fix    = use_llm_fix and (ollama_client is not None)

    def validate_and_fix(
        self,
        sql:    str,
        schema: Optional[str] = None,
    ) -> Dict:
        """
        Valide et corrige une requête SQL.

        Args:
            sql:    Requête SQL à valider.
            schema: Contexte du schéma (optionnel, pour la correction LLM).

        Returns:
            Dictionnaire :
            {
                "valid":     bool,
                "fixed_sql": str,
                "errors":    List[str],
                "warnings":  List[str],
                "fixes":     List[str],
            }
        """
        if not sql or not sql.strip():
            return {
                "valid":     False,
                "fixed_sql": "",
                "errors":    ["Requête SQL vide"],
                "warnings":  [],
                "fixes":     [],
            }

        errors   = []
        warnings = []
        fixes    = []
        current_sql = sql.strip()

        # ── Étape 1 : Corrections automatiques par regex ───────────────────
        current_sql, auto_fixes = self._apply_auto_corrections(current_sql)
        fixes.extend(auto_fixes)

        # ── Étape 2 : Vérification syntaxique ─────────────────────────────
        syntax_errors = self._check_syntax(current_sql)
        errors.extend(syntax_errors)

        # ── Étape 3 : Vérification des tables ─────────────────────────────
        table_errors, table_warnings = self._check_tables(current_sql)
        errors.extend(table_errors)
        warnings.extend(table_warnings)

        # ── Étape 4 : Vérification colonnes virtuelles ─────────────────────
        virtual_errors = self._check_virtual_columns(current_sql)
        errors.extend(virtual_errors)

        # ── Étape 5 : Vérification DELETE sans WHERE ───────────────────────
        delete_warnings = self._check_dangerous_operations(current_sql)
        warnings.extend(delete_warnings)

        # ── Étape 6 : Correction LLM si erreurs persistantes ──────────────
        if errors and self.use_llm_fix and self.client:
            logger.info("Tentative de correction LLM — %d erreur(s)", len(errors))
            fixed, llm_fixes = self._fix_with_llm(current_sql, errors, schema)
            if fixed:
                current_sql = fixed
                fixes.extend(llm_fixes)
                # Re-valider après correction LLM
                errors = []
                errors.extend(self._check_syntax(current_sql))
                table_errs, _ = self._check_tables(current_sql)
                errors.extend(table_errs)
                errors.extend(self._check_virtual_columns(current_sql))

        # ── Résultat final ─────────────────────────────────────────────────
        is_valid = len(errors) == 0

        if is_valid:
            logger.debug("SQL validé avec succès — %d fix(es) appliqué(s)", len(fixes))
        else:
            logger.warning("SQL invalide — %d erreur(s) : %s", len(errors), errors)

        return {
            "valid":     is_valid,
            "fixed_sql": current_sql,
            "errors":    errors,
            "warnings":  warnings,
            "fixes":     fixes,
        }

    # ── Corrections automatiques ──────────────────────────────────────────────

    @staticmethod
    def _apply_auto_corrections(sql: str) -> Tuple[str, List[str]]:
        """
        Applique les corrections automatiques par regex.

        Returns:
            (sql_corrigé, liste_des_corrections_appliquées)
        """
        fixes = []
        current = sql

        for pattern, replacement in AUTO_CORRECTIONS:
            new_sql = re.sub(pattern, replacement, current, flags=re.IGNORECASE)
            if new_sql != current:
                fixes.append(f"Auto-correction : {pattern!r} → {replacement!r}")
                current = new_sql

        # Correction spécifique : ajouter point-virgule manquant
        if current.strip() and not current.strip().endswith(";"):
            current = current.strip() + ";"
            fixes.append("Ajout du point-virgule final")

        return current, fixes

    # ── Vérifications ─────────────────────────────────────────────────────────

    @staticmethod
    def _check_syntax(sql: str) -> List[str]:
        """
        Vérifie la syntaxe SQL basique.

        Returns:
            Liste des erreurs de syntaxe.
        """
        errors = []
        upper  = sql.upper().strip()

        # Vérifier qu'il y a un mot-clé SQL principal
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE"]
        if not any(upper.startswith(kw) for kw in sql_keywords):
            errors.append(
                f"La requête ne commence pas par un mot-clé SQL valide "
                f"({', '.join(sql_keywords)})"
            )

        # Vérifier les parenthèses équilibrées
        open_count  = sql.count("(")
        close_count = sql.count(")")
        if open_count != close_count:
            errors.append(
                f"Parenthèses non équilibrées : {open_count} ouvrantes, {close_count} fermantes"
            )

        # Vérifier les guillemets équilibrés (simples)
        # Compter les guillemets simples non échappés
        single_quotes = len(re.findall(r"(?<!\\)'", sql))
        if single_quotes % 2 != 0:
            errors.append("Guillemets simples non équilibrés")

        # Vérifier SELECT sans FROM (sauf SELECT 1, SELECT NOW(), etc.)
        if upper.startswith("SELECT") and "FROM" not in upper:
            # Autoriser SELECT sans FROM pour les expressions simples
            if not re.search(r"SELECT\s+[\d\w\(\)'\"]+\s*;?\s*$", sql, re.IGNORECASE):
                errors.append("SELECT sans clause FROM")

        # Vérifier UPDATE sans WHERE
        if upper.startswith("UPDATE") and "WHERE" not in upper:
            errors.append("UPDATE sans clause WHERE (dangereux)")

        return errors

    @staticmethod
    def _check_tables(sql: str) -> Tuple[List[str], List[str]]:
        """
        Vérifie que les tables référencées existent dans le schéma.

        Returns:
            (erreurs, avertissements)
        """
        errors   = []
        warnings = []

        # Extraire les noms de tables depuis FROM et JOIN
        table_pattern = re.compile(
            r"(?:FROM|JOIN)\s+`?(\w+)`?(?:\s+(?:AS\s+)?`?\w+`?)?",
            re.IGNORECASE,
        )
        found_tables = table_pattern.findall(sql)

        for table in found_tables:
            # Ignorer les sous-requêtes et alias
            if table.upper() in ("SELECT", "WHERE", "ON", "AND", "OR"):
                continue
            if table not in VALID_TABLES:
                # Vérifier si c'est une casse incorrecte
                lower_map = {t.lower(): t for t in VALID_TABLES}
                if table.lower() in lower_map:
                    errors.append(
                        f"Casse incorrecte pour la table '{table}'. "
                        f"Utiliser '{lower_map[table.lower()]}'"
                    )
                else:
                    errors.append(f"Table '{table}' introuvable dans le schéma")

        return errors, warnings

    @staticmethod
    def _check_virtual_columns(sql: str) -> List[str]:
        """
        Vérifie que les colonnes virtuelles ne sont pas utilisées dans INSERT/UPDATE.

        Returns:
            Liste des erreurs.
        """
        errors = []
        upper  = sql.upper()

        if not ("INSERT" in upper or "UPDATE" in upper):
            return errors

        for table, column in VIRTUAL_COLUMNS:
            # Chercher la colonne dans un contexte INSERT/UPDATE
            pattern = re.compile(
                rf"(?:INSERT\s+INTO\s+`?{table}`?|UPDATE\s+`?{table}`?)"
                rf".*?`?{column}`?",
                re.IGNORECASE | re.DOTALL,
            )
            if pattern.search(sql):
                errors.append(
                    f"La colonne '{column}' de la table '{table}' est VIRTUELLE "
                    f"et ne peut pas être utilisée dans INSERT/UPDATE"
                )

        return errors

    @staticmethod
    def _check_dangerous_operations(sql: str) -> List[str]:
        """
        Détecte les opérations potentiellement dangereuses.

        Returns:
            Liste des avertissements.
        """
        warnings = []
        upper    = sql.upper().strip()

        # DELETE sans WHERE
        if upper.startswith("DELETE") and "WHERE" not in upper:
            warnings.append("DELETE sans clause WHERE — suppression de toutes les lignes")

        # SELECT * (peut retourner trop de données)
        if "SELECT *" in upper or "SELECT\n*" in upper:
            warnings.append("SELECT * — préférer lister les colonnes explicitement")

        # Pas de LIMIT sur un SELECT potentiellement large
        if upper.startswith("SELECT") and "LIMIT" not in upper and "COUNT" not in upper:
            warnings.append("SELECT sans LIMIT — peut retourner un grand nombre de lignes")

        return warnings

    # ── Correction LLM ────────────────────────────────────────────────────────

    def _fix_with_llm(
        self,
        sql:    str,
        errors: List[str],
        schema: Optional[str],
    ) -> Tuple[Optional[str], List[str]]:
        """
        Tente de corriger le SQL via SQLCoder.

        Args:
            sql:    SQL à corriger.
            errors: Liste des erreurs détectées.
            schema: Contexte du schéma (optionnel).

        Returns:
            (sql_corrigé, liste_des_corrections) ou (None, []) si échec.
        """
        errors_str = "\n".join(f"- {e}" for e in errors)
        schema_str = f"\n\nSCHÉMA:\n{schema}" if schema else ""

        prompt = (
            f"Corrige cette requête SQL MySQL qui contient des erreurs :{schema_str}\n\n"
            f"SQL INCORRECT:\n{sql}\n\n"
            f"ERREURS DÉTECTÉES:\n{errors_str}\n\n"
            f"SQL CORRIGÉ (retourner UNIQUEMENT le SQL):"
        )

        try:
            from ai.intent_agent import OllamaUnavailableError
            raw = self.client.generate(
                model=self.sqlcoder_model,
                prompt=prompt,
                system=SYSTEM_SQLCODER_FIX,
                temperature=0.0,
            )

            # Extraire le SQL de la réponse
            from ai.sql_generator import SQLGenerator
            fixed_sql = SQLGenerator._extract_sql(raw)

            if fixed_sql and fixed_sql != sql:
                return fixed_sql, [f"Correction LLM ({self.sqlcoder_model}) appliquée"]
            return None, []

        except Exception as exc:
            logger.warning("Correction LLM échouée : %s", str(exc))
            return None, []
