"""
Table Matcher — Healthcare AI Platform
Combine les scores FAISS et les prédictions Llama pour identifier
les tables SQL les plus pertinentes pour une requête.

Score hybride : 0.6 × faiss_score + 0.4 × llama_confidence

Architecture :
  TableMatcher  → orchestrateur principal
  HybridScorer  → calcul du score hybride FAISS + LLM
"""

import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Pondération du score hybride (doit sommer à 1.0)
FAISS_WEIGHT = 0.6
LLAMA_WEIGHT = 0.4

# Score minimum pour inclure une table dans les résultats
MIN_HYBRID_SCORE = 0.25

# Nombre maximum de tables à retourner
MAX_TABLES = 4

# Tables valides du schéma
VALID_TABLES = {
    "Patient", "Medical_records", "Consultation",
    "Medical_staff", "Service", "Users", "AI_Query_Logs",
}

# Exemples de données par table (pour enrichir le contexte Qwen)
TABLE_SAMPLE_DATA = {
    "Service": [
        {"id_service": 1, "service_name": "Cardiologie"},
        {"id_service": 2, "service_name": "Neurologie"},
        {"id_service": 3, "service_name": "Oncologie"},
    ],
    "Medical_staff": [
        {"id_staff": 1, "name_staff": "Dr. Martin", "position_staff": "Doctor",
         "speciality": "Cardiologie", "id_service": 1},
        {"id_staff": 2, "name_staff": "Inf. Dupont", "position_staff": "Nurse",
         "speciality": None, "id_service": 1},
    ],
    "Patient": [
        {"id_patient": 1, "first_name": "Jean", "last_name": "Dupont",
         "birthdate": "1985-03-15", "gender": "Male", "age": 39},
        {"id_patient": 2, "first_name": "Marie", "last_name": "Martin",
         "birthdate": "1972-07-22", "gender": "Female", "age": 52},
    ],
    "Medical_records": [
        {"id_record": 1, "id_patient": 1, "blood_group": "A+",
         "allergies": "Pénicilline", "chronic_diseases": "Diabète type 2"},
        {"id_record": 2, "id_patient": 2, "blood_group": "O-",
         "allergies": None, "chronic_diseases": "Hypertension"},
    ],
    "Consultation": [
        {"id_consultation": 1, "diagnosis": "Hypertension artérielle",
         "treatment": "Amlodipine 5mg", "date": "2024-01-15 10:30:00",
         "id_staff": 1, "id_patient": 1},
        {"id_consultation": 2, "diagnosis": "Diabète type 2 décompensé",
         "treatment": "Insuline + régime", "date": "2024-01-16 14:00:00",
         "id_staff": 1, "id_patient": 2},
    ],
    "Users": [
        {"id_user": 1, "username": "admin", "role": "admin", "is_active": True},
        {"id_user": 2, "username": "dr.martin", "role": "doctor", "is_active": True},
    ],
    "AI_Query_Logs": [
        {"id_log": 1, "prompt": "Liste les patients diabétiques",
         "detected_intent": "READ_ONLY", "confidence_score": 0.92, "latency_ms": 1250},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  HybridScorer — calcul du score hybride
# ─────────────────────────────────────────────────────────────────────────────

class HybridScorer:
    """
    Calcule le score hybride combinant FAISS et prédiction LLM.

    Score hybride = faiss_weight × faiss_score + llama_weight × llama_score
    """

    def __init__(
        self,
        faiss_weight: float = FAISS_WEIGHT,
        llama_weight: float = LLAMA_WEIGHT,
    ):
        """
        Args:
            faiss_weight: Poids du score FAISS (défaut 0.6).
            llama_weight: Poids du score LLM (défaut 0.4).
        """
        if abs(faiss_weight + llama_weight - 1.0) > 1e-6:
            raise ValueError(
                f"faiss_weight + llama_weight doit être égal à 1.0 "
                f"(reçu {faiss_weight + llama_weight})"
            )
        self.faiss_weight = faiss_weight
        self.llama_weight = llama_weight

    def compute(
        self,
        table_name:       str,
        faiss_score:      float,
        llama_prediction: Dict,
    ) -> float:
        """
        Calcule le score hybride pour une table.

        Args:
            table_name:       Nom de la table.
            faiss_score:      Score de similarité FAISS (0-1).
            llama_prediction: Résultat de l'IntentAgent avec tables et confidence.

        Returns:
            Score hybride entre 0 et 1.
        """
        # Score LLM : 1.0 si la table est dans les prédictions Llama, 0 sinon
        # Pondéré par la confiance globale du LLM
        llama_tables     = llama_prediction.get("tables", [])
        llama_confidence = float(llama_prediction.get("confidence", 0.5))

        if table_name in llama_tables:
            llama_score = llama_confidence
        else:
            llama_score = 0.0

        hybrid = (self.faiss_weight * faiss_score) + (self.llama_weight * llama_score)
        return round(min(1.0, max(0.0, hybrid)), 4)

    def rank_tables(
        self,
        faiss_results:    List[Dict],
        llama_prediction: Dict,
        min_score:        float = MIN_HYBRID_SCORE,
        max_tables:       int   = MAX_TABLES,
    ) -> List[Dict]:
        """
        Classe les tables par score hybride décroissant.

        Args:
            faiss_results:    Résultats FAISS [{table, score, ...}].
            llama_prediction: Prédiction LLM {tables, confidence, ...}.
            min_score:        Score minimum pour inclure une table.
            max_tables:       Nombre maximum de tables à retourner.

        Returns:
            Liste de tables classées avec scores hybrides.
        """
        # Agréger les scores FAISS par table (prendre le max)
        faiss_by_table: Dict[str, float] = {}
        for result in faiss_results:
            table = result["table"]
            score = result["score"]
            if table not in faiss_by_table or score > faiss_by_table[table]:
                faiss_by_table[table] = score

        # Normaliser les scores FAISS (0-1)
        if faiss_by_table:
            max_faiss = max(faiss_by_table.values())
            if max_faiss > 0:
                faiss_by_table = {t: s / max_faiss for t, s in faiss_by_table.items()}

        # Ajouter les tables Llama non présentes dans FAISS avec score 0
        for table in llama_prediction.get("tables", []):
            if table in VALID_TABLES and table not in faiss_by_table:
                faiss_by_table[table] = 0.0

        # Calculer les scores hybrides
        ranked = []
        for table, faiss_score in faiss_by_table.items():
            hybrid_score = self.compute(table, faiss_score, llama_prediction)
            if hybrid_score >= min_score:
                ranked.append({
                    "table":        table,
                    "hybrid_score": hybrid_score,
                    "faiss_score":  round(faiss_score, 4),
                    "llama_score":  round(
                        float(llama_prediction.get("confidence", 0))
                        if table in llama_prediction.get("tables", []) else 0.0,
                        4,
                    ),
                    "in_llama":     table in llama_prediction.get("tables", []),
                })

        # Trier par score hybride décroissant
        ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:max_tables]


# ─────────────────────────────────────────────────────────────────────────────
#  TableMatcher — orchestrateur principal
# ─────────────────────────────────────────────────────────────────────────────

class TableMatcher:
    """
    Identifie les tables SQL pertinentes pour une requête en langage naturel.
    Combine la recherche sémantique FAISS et les prédictions de l'IntentAgent.

    Usage:
        matcher = TableMatcher(faiss_index, embedder)
        tables  = matcher.match_tables("patients diabétiques", llama_result)
        context = matcher.get_schema_context(tables)
    """

    def __init__(
        self,
        faiss_index,
        embedder,
        faiss_weight: float = FAISS_WEIGHT,
        llama_weight: float = LLAMA_WEIGHT,
        top_k_faiss:  int   = 10,
    ):
        """
        Args:
            faiss_index:  Instance FAISSTableIndex.
            embedder:     Instance BiomedicalEmbedder.
            faiss_weight: Poids du score FAISS dans le score hybride.
            llama_weight: Poids du score LLM dans le score hybride.
            top_k_faiss:  Nombre de résultats FAISS à récupérer avant fusion.
        """
        self.faiss_index  = faiss_index
        self.embedder     = embedder
        self.scorer       = HybridScorer(faiss_weight, llama_weight)
        self.top_k_faiss  = top_k_faiss

        from database.schema_builder import schema_builder
        self._schema_builder = schema_builder

    # ── Méthode principale ────────────────────────────────────────────────────

    def match_tables(
        self,
        prompt:           str,
        llama_prediction: Dict,
        max_tables:       int   = MAX_TABLES,
        min_score:        float = MIN_HYBRID_SCORE,
    ) -> List[str]:
        """
        Identifie les tables les plus pertinentes pour une requête.
        Combine FAISS (similarité sémantique) et Llama (prédiction LLM).

        Args:
            prompt:           Requête en langage naturel.
            llama_prediction: Résultat de l'IntentAgent.analyze().
            max_tables:       Nombre maximum de tables à retourner.
            min_score:        Score hybride minimum pour inclure une table.

        Returns:
            Liste de noms de tables triés par pertinence décroissante.
        """
        if not prompt or not prompt.strip():
            return llama_prediction.get("tables", ["Patient"])[:max_tables]

        # ── Recherche FAISS ────────────────────────────────────────────────
        faiss_results = []
        if self.faiss_index.is_built:
            try:
                faiss_results = self.faiss_index.search(
                    prompt.strip(),
                    top_k=self.top_k_faiss,
                )
            except Exception as exc:
                logger.warning("Erreur FAISS search : %s — utilisation Llama seul", str(exc))

        # ── Score hybride ──────────────────────────────────────────────────
        if faiss_results:
            ranked = self.scorer.rank_tables(
                faiss_results=faiss_results,
                llama_prediction=llama_prediction,
                min_score=min_score,
                max_tables=max_tables,
            )
            tables = [r["table"] for r in ranked]

            logger.debug(
                "Table matching — prompt=%r tables=%s scores=%s",
                prompt[:50],
                tables,
                [r["hybrid_score"] for r in ranked],
            )
        else:
            # Fallback : utiliser uniquement les prédictions Llama
            tables = [
                t for t in llama_prediction.get("tables", ["Patient"])
                if t in VALID_TABLES
            ][:max_tables]
            logger.debug("Table matching (Llama only) — tables=%s", tables)

        # Garantir au moins une table
        if not tables:
            tables = ["Patient"]

        return tables

    def match_tables_detailed(
        self,
        prompt:           str,
        llama_prediction: Dict,
        max_tables:       int   = MAX_TABLES,
    ) -> List[Dict]:
        """
        Version détaillée de match_tables avec scores inclus.

        Returns:
            Liste de dicts avec table, hybrid_score, faiss_score, llama_score.
        """
        if not self.faiss_index.is_built:
            return [
                {"table": t, "hybrid_score": 0.5, "faiss_score": 0.0, "llama_score": 0.5}
                for t in llama_prediction.get("tables", ["Patient"])[:max_tables]
            ]

        faiss_results = self.faiss_index.search(prompt.strip(), top_k=self.top_k_faiss)
        return self.scorer.rank_tables(
            faiss_results=faiss_results,
            llama_prediction=llama_prediction,
            max_tables=max_tables,
        )

    # ── Contexte schéma pour Qwen ─────────────────────────────────────────────

    def get_schema_context(
        self,
        tables:          List[str],
        include_samples: bool = True,
        include_joins:   bool = True,
    ) -> str:
        """
        Génère le contexte schéma SQL optimisé pour le prompt Qwen Text2SQL.
        Inclut les définitions de tables, les relations FK et des exemples de données.

        Args:
            tables:          Liste des tables à inclure.
            include_samples: Inclure des exemples de données (quelques lignes).
            include_joins:   Inclure les chemins de jointure disponibles.

        Returns:
            Chaîne formatée prête à être injectée dans le prompt Qwen.
        """
        lines = [
            "-- ============================================================",
            "-- DATABASE: healthcare_ai_platform (MySQL 8.0, utf8mb4)",
            "-- ============================================================",
            "",
        ]

        # ── Définitions des tables ─────────────────────────────────────────
        for table in tables:
            try:
                schema = self._schema_builder.get_table_schema(table)
            except KeyError:
                continue

            from embeddings.faiss_index import TABLE_DESCRIPTIONS, COLUMN_DESCRIPTIONS
            table_desc   = TABLE_DESCRIPTIONS.get(table, schema.get("description", ""))
            col_descs    = COLUMN_DESCRIPTIONS.get(table, {})

            lines.append(f"-- {table_desc.split('.')[0]}")
            lines.append(f"CREATE TABLE `{table}` (")

            col_lines = []
            for col_name, col_info in schema["columns"].items():
                col_type    = col_info["type"]
                pk_str      = " PRIMARY KEY AUTO_INCREMENT" if col_info.get("pk") else ""
                null_str    = " NOT NULL" if not col_info.get("nullable", True) else ""
                unique_str  = " UNIQUE" if col_info.get("unique") else ""
                virtual_str = " GENERATED ALWAYS AS (...) VIRTUAL" if "VIRTUAL" in col_type else ""
                col_type_clean = col_type.replace(" VIRTUAL", "")

                # Commentaire avec description sémantique
                col_comment = col_descs.get(col_name, "")
                comment_str = f"  -- {col_comment}" if col_comment else ""

                col_lines.append(
                    f"    `{col_name}` {col_type_clean}{pk_str}{null_str}{unique_str}{virtual_str}{comment_str}"
                )

            lines.append(",\n".join(col_lines))
            lines.append(");\n")

        # ── Jointures disponibles ──────────────────────────────────────────
        if include_joins and len(tables) > 1:
            join_lines = []
            for i, table_a in enumerate(tables):
                for table_b in tables[i + 1:]:
                    join_path = self._schema_builder.get_join_path(table_a, table_b)
                    if join_path:
                        join_lines.append(
                            f"-- JOIN {table_a} ↔ {table_b}: {join_path}"
                        )
            if join_lines:
                lines.append("-- ── Jointures disponibles ──────────────────────────────")
                lines.extend(join_lines)
                lines.append("")

        # ── Exemples de données ────────────────────────────────────────────
        if include_samples:
            lines.append("-- ── Exemples de données (pour référence) ───────────────")
            for table in tables:
                samples = TABLE_SAMPLE_DATA.get(table, [])
                if samples:
                    lines.append(f"-- {table} (exemples) :")
                    for row in samples[:2]:  # Max 2 exemples par table
                        row_str = ", ".join(f"{k}={repr(v)}" for k, v in list(row.items())[:4])
                        lines.append(f"--   {row_str}")
            lines.append("")

        # ── Règles SQL importantes ─────────────────────────────────────────
        lines.extend([
            "-- ── Règles importantes ─────────────────────────────────────",
            "-- 1. La colonne `age` dans Patient est VIRTUELLE — ne pas utiliser dans INSERT/UPDATE",
            "-- 2. Noms complets : CONCAT(first_name, ' ', last_name)",
            "-- 3. Respecter la casse des noms de tables : Patient, Medical_staff, etc.",
            "-- 4. Utiliser utf8mb4 pour les comparaisons de chaînes",
            "-- 5. Les dates de consultation sont dans la colonne `date` (DATETIME)",
        ])

        return "\n".join(lines)

    # ── Chemins de jointure ───────────────────────────────────────────────────

    def get_join_paths(self, table_a: str, table_b: str) -> List[str]:
        """
        Retourne les chemins de jointure entre deux tables.
        Inclut les jointures directes et indirectes (via table intermédiaire).

        Args:
            table_a: Première table.
            table_b: Deuxième table.

        Returns:
            Liste de conditions JOIN SQL.
        """
        paths = []

        # Jointure directe
        direct = self._schema_builder.get_join_path(table_a, table_b)
        if direct:
            paths.append(direct)

        # Jointures indirectes (via une table intermédiaire)
        all_tables = self._schema_builder.get_all_tables()
        for intermediate in all_tables:
            if intermediate in (table_a, table_b):
                continue
            path_a = self._schema_builder.get_join_path(table_a, intermediate)
            path_b = self._schema_builder.get_join_path(intermediate, table_b)
            if path_a and path_b:
                paths.append(f"{path_a} AND {path_b} (via {intermediate})")

        return paths

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Retourne le statut du TableMatcher."""
        return {
            "faiss_built":   self.faiss_index.is_built,
            "faiss_entries": self.faiss_index.entry_count if self.faiss_index.is_built else 0,
            "embedder_model": self.embedder.model_name,
            "faiss_weight":  self.scorer.faiss_weight,
            "llama_weight":  self.scorer.llama_weight,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_table_matcher(
    index_path:    str = "embeddings/faiss_store/healthcare.index",
    metadata_path: str = "embeddings/faiss_store/metadata.json",
    model_key:     str = "biolord",
    faiss_weight:  float = FAISS_WEIGHT,
    llama_weight:  float = LLAMA_WEIGHT,
) -> TableMatcher:
    """
    Crée un TableMatcher complet avec embedder et index FAISS.
    Tente de charger l'index existant, sinon le construit.

    Args:
        index_path:    Chemin de l'index FAISS.
        metadata_path: Chemin des métadonnées.
        model_key:     Modèle d'embedding ('biolord', 'biobert', 'minilm').
        faiss_weight:  Poids FAISS dans le score hybride.
        llama_weight:  Poids LLM dans le score hybride.

    Returns:
        TableMatcher configuré et prêt à l'emploi.
    """
    from embeddings.biomedical_embeddings import create_embedder
    from embeddings.faiss_index import FAISSTableIndex

    embedder    = create_embedder(model_key=model_key)
    faiss_index = FAISSTableIndex(
        embedder=embedder,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    # Tenter de charger l'index existant
    loaded = faiss_index.load_index()
    if not loaded:
        logger.info("Index FAISS non trouvé — construction en cours...")
        faiss_index.build_index()
        faiss_index.save_index()

    return TableMatcher(
        faiss_index=faiss_index,
        embedder=embedder,
        faiss_weight=faiss_weight,
        llama_weight=llama_weight,
    )
