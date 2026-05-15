"""
Index FAISS pour le Table Matching — Healthcare AI Platform
Indexe toutes les tables et colonnes du schéma DB avec leurs embeddings biomédicaux.

Architecture :
  FAISSTableIndex  → classe principale (build, search, save, load)
  IndexEntry       → entrée dans l'index (table, colonne, description, embedding)

Chaque entrée FAISS représente :
  - Une table entière (description sémantique)
  - Une colonne spécifique (nom + description + tags sémantiques)
"""

import os
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  IndexEntry — entrée dans l'index FAISS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndexEntry:
    """
    Représente une entrée dans l'index FAISS.
    Peut être une table entière ou une colonne spécifique.
    """
    entry_id:    int          # Index dans le vecteur FAISS
    table_name:  str          # Nom de la table MySQL
    column_name: Optional[str]  # Nom de la colonne (None si entrée de table)
    description: str          # Description sémantique en langage naturel
    entry_type:  str          # "table" ou "column"
    tags:        List[str]    # Tags sémantiques supplémentaires

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "IndexEntry":
        return cls(**data)


# ─────────────────────────────────────────────────────────────────────────────
#  Descriptions sémantiques médicales enrichies
# ─────────────────────────────────────────────────────────────────────────────

# Descriptions des tables en langage naturel (FR + EN pour meilleur matching)
TABLE_DESCRIPTIONS = {
    "Service": (
        "Service hospitalier ou département médical. "
        "Contient les unités de soins comme la Cardiologie, Neurologie, Oncologie, Pédiatrie, Urgences. "
        "Hospital department, medical ward, clinical unit, specialty service."
    ),
    "Medical_staff": (
        "Personnel médical de l'hôpital : médecins, infirmiers, techniciens, administrateurs. "
        "Contient le nom, la spécialité médicale, le poste et le service d'appartenance. "
        "Doctor, physician, nurse, medical staff, healthcare worker, clinician, specialist."
    ),
    "Patient": (
        "Informations personnelles et démographiques des patients hospitalisés. "
        "Contient le prénom, nom, date de naissance, âge, genre, contact. "
        "Patient demographics, personal information, age, gender, date of birth."
    ),
    "Medical_records": (
        "Dossier médical complet d'un patient : allergies, maladies chroniques, groupe sanguin, historique médical. "
        "Contient les antécédents médicaux, pathologies, traitements passés. "
        "Medical record, health record, allergies, chronic diseases, blood type, medical history, patient file."
    ),
    "Consultation": (
        "Consultation médicale entre un patient et un médecin. "
        "Contient le diagnostic, le traitement prescrit, le rapport médical et la date. "
        "Medical consultation, appointment, diagnosis, treatment, medical report, clinical visit."
    ),
    "Users": (
        "Comptes utilisateurs du système avec rôles d'accès (admin, médecin, staff, patient). "
        "Gestion des authentifications et permissions RBAC. "
        "User account, login, role, permission, access control, authentication."
    ),
    "AI_Query_Logs": (
        "Historique des requêtes IA Text2SQL : prompts utilisateurs, SQL généré, métriques de performance. "
        "Logs d'évaluation du système d'intelligence artificielle. "
        "AI query log, SQL generation, prompt history, performance metrics, evaluation."
    ),
}

# Descriptions sémantiques des colonnes par table
COLUMN_DESCRIPTIONS = {
    "Service": {
        "id_service":   "Identifiant unique du service hospitalier",
        "service_name": "Nom du service médical (Cardiologie, Neurologie, Oncologie, Urgences, Pédiatrie)",
    },
    "Medical_staff": {
        "id_staff":       "Identifiant unique du membre du personnel médical",
        "name_staff":     "Nom complet du médecin, infirmier ou technicien",
        "position_staff": "Poste ou fonction : Doctor (médecin), Nurse (infirmier), Technician, Administrator",
        "speciality":     "Spécialité médicale du médecin (cardiologie, neurologie, chirurgie, etc.)",
        "id_service":     "Service hospitalier d'appartenance du personnel médical",
        "email":          "Adresse email professionnelle du personnel médical",
        "phone":          "Numéro de téléphone du personnel médical",
        "created_at":     "Date d'enregistrement du membre du personnel dans le système",
    },
    "Patient": {
        "id_patient": "Identifiant unique du patient",
        "first_name": "Prénom du patient",
        "last_name":  "Nom de famille du patient",
        "birthdate":  "Date de naissance du patient (format DATE)",
        "gender":     "Genre du patient : Male (homme) ou Female (femme)",
        "age":        "Âge calculé automatiquement du patient en années (colonne virtuelle MySQL)",
        "email":      "Adresse email du patient pour les communications",
        "phone":      "Numéro de téléphone du patient",
        "created_at": "Date d'admission ou d'enregistrement du patient",
    },
    "Medical_records": {
        "id_record":        "Identifiant unique du dossier médical",
        "id_patient":       "Référence au patient propriétaire du dossier médical",
        "allergies":        "Liste des allergies connues du patient (médicaments, aliments, substances)",
        "chronic_diseases": "Maladies chroniques du patient : diabète, hypertension, asthme, cancer, BPCO, insuffisance rénale",
        "blood_group":      "Groupe sanguin du patient : A+, A-, B+, B-, AB+, AB-, O+, O-",
        "medical_history":  "Historique médical complet : antécédents, chirurgies, hospitalisations passées",
        "last_updated":     "Date de dernière mise à jour du dossier médical",
    },
    "Consultation": {
        "id_consultation": "Identifiant unique de la consultation médicale",
        "diagnosis":       "Diagnostic médical établi lors de la consultation",
        "treatment":       "Traitement prescrit : médicaments, thérapies, interventions",
        "medical_report":  "Rapport médical détaillé de la consultation",
        "date":            "Date et heure de la consultation médicale",
        "id_staff":        "Médecin ou personnel médical ayant effectué la consultation",
        "id_patient":      "Patient concerné par la consultation médicale",
    },
    "Users": {
        "id_user":       "Identifiant unique du compte utilisateur",
        "username":      "Nom d'utilisateur pour la connexion au système",
        "email":         "Adresse email du compte utilisateur",
        "password_hash": "Mot de passe hashé (bcrypt) — jamais exposé",
        "role":          "Rôle RBAC : admin, doctor (médecin), staff (personnel), patient",
        "id_staff":      "Lien vers le profil du personnel médical si l'utilisateur est un médecin",
        "id_patient":    "Lien vers le profil patient si l'utilisateur est un patient",
        "is_active":     "Statut du compte : actif (True) ou désactivé (False)",
        "created_at":    "Date de création du compte utilisateur",
    },
    "AI_Query_Logs": {
        "id_log":           "Identifiant unique du log de requête IA",
        "user_id":          "Utilisateur ayant soumis la requête IA",
        "prompt":           "Requête en langage naturel soumise au système IA",
        "detected_intent":  "Intention détectée : READ_ONLY ou READ_WRITE",
        "detected_tables":  "Tables identifiées par le pipeline IA (JSON)",
        "generated_sql":    "Requête SQL générée par le modèle Text2SQL",
        "execution_result": "Résultat de l'exécution de la requête SQL",
        "exact_match":      "Indicateur si le SQL généré correspond exactement au SQL de référence",
        "confidence_score": "Score de confiance du modèle IA (0.0 à 1.0)",
        "latency_ms":       "Temps de traitement de la requête en millisecondes",
        "created_at":       "Horodatage de la requête IA",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  FAISSTableIndex — index principal
# ─────────────────────────────────────────────────────────────────────────────

class FAISSTableIndex:
    """
    Index FAISS pour la recherche sémantique de tables et colonnes.

    Chaque entrée de l'index correspond à :
      - Une table (description globale)
      - Une colonne (description + tags sémantiques)

    La recherche retourne les tables/colonnes les plus pertinentes
    pour une requête en langage naturel.

    Usage:
        embedder = BiomedicalEmbedder()
        index = FAISSTableIndex(embedder)
        index.build_index()
        results = index.search("patients diabétiques", top_k=5)
    """

    def __init__(
        self,
        embedder,
        index_path:    str = "embeddings/faiss_store/healthcare.index",
        metadata_path: str = "embeddings/faiss_store/metadata.json",
    ):
        """
        Args:
            embedder:      Instance BiomedicalEmbedder pour générer les embeddings.
            index_path:    Chemin de sauvegarde de l'index FAISS binaire.
            metadata_path: Chemin de sauvegarde des métadonnées JSON.
        """
        self.embedder      = embedder
        self.index_path    = index_path
        self.metadata_path = metadata_path

        self._index:   Optional[object]     = None   # Index FAISS
        self._entries: List[IndexEntry]     = []     # Métadonnées
        self._is_built = False

    # ── Construction de l'index ───────────────────────────────────────────────

    def build_index(self, schema: Optional[Dict] = None) -> None:
        """
        Construit l'index FAISS à partir du schéma de la base de données.
        Indexe chaque table et chaque colonne avec leurs descriptions sémantiques.

        Args:
            schema: Schéma optionnel (dict). Si None, utilise le schéma statique.
        """
        import faiss

        if schema is None:
            from database.schema_builder import SCHEMA as schema

        logger.info("Construction de l'index FAISS — %d tables", len(schema))
        start_time = time.perf_counter()

        entries  = []
        texts    = []
        entry_id = 0

        for table_name, table_info in schema.items():
            # ── Entrée de table ────────────────────────────────────────────
            table_desc = TABLE_DESCRIPTIONS.get(table_name, table_info.get("description", table_name))
            tags       = table_info.get("semantic_tags", [])

            # Texte enrichi pour l'embedding de la table
            table_text = self._build_table_text(table_name, table_desc, tags)

            entries.append(IndexEntry(
                entry_id=entry_id,
                table_name=table_name,
                column_name=None,
                description=table_desc,
                entry_type="table",
                tags=tags,
            ))
            texts.append(table_text)
            entry_id += 1

            # ── Entrées de colonnes ────────────────────────────────────────
            col_descriptions = COLUMN_DESCRIPTIONS.get(table_name, {})
            for col_name, col_info in table_info.get("columns", {}).items():
                col_desc = col_descriptions.get(col_name, f"Colonne {col_name} de la table {table_name}")

                # Texte enrichi pour l'embedding de la colonne
                col_text = self._build_column_text(table_name, col_name, col_desc, col_info)

                entries.append(IndexEntry(
                    entry_id=entry_id,
                    table_name=table_name,
                    column_name=col_name,
                    description=col_desc,
                    entry_type="column",
                    tags=[col_name, table_name],
                ))
                texts.append(col_text)
                entry_id += 1

        logger.info("Génération de %d embeddings...", len(texts))

        # Générer tous les embeddings en batch
        embeddings = self.embedder.embed_batch(texts, batch_size=32, show_progress=True)

        # Vérifier la dimension
        dimension = embeddings.shape[1]
        logger.info("Dimension des embeddings : %d", dimension)

        # Construire l'index FAISS (IndexFlatIP = produit scalaire = cosinus si normalisé)
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings.astype(np.float32))

        self._entries  = entries
        self._is_built = True

        elapsed = round((time.perf_counter() - start_time) * 1000, 1)
        logger.info(
            "Index FAISS construit — %d entrées, dimension=%d, temps=%dms",
            len(entries), dimension, elapsed,
        )

    @staticmethod
    def _build_table_text(table_name: str, description: str, tags: List[str]) -> str:
        """Construit le texte d'embedding pour une table."""
        tags_str = " ".join(tags) if tags else ""
        return f"{table_name}: {description}. Keywords: {tags_str}"

    @staticmethod
    def _build_column_text(
        table_name: str,
        col_name:   str,
        description: str,
        col_info:   Dict,
    ) -> str:
        """Construit le texte d'embedding pour une colonne."""
        col_type = col_info.get("type", "")
        values   = col_info.get("values", [])
        values_str = f" Values: {', '.join(values)}" if values else ""
        return (
            f"{table_name}.{col_name}: {description}. "
            f"Type: {col_type}.{values_str} "
            f"Table: {table_name}"
        )

    # ── Recherche ─────────────────────────────────────────────────────────────

    def search(
        self,
        query:  str,
        top_k:  int = 5,
        entry_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Recherche les tables/colonnes les plus pertinentes pour une requête.

        Args:
            query:      Requête en langage naturel.
            top_k:      Nombre de résultats à retourner.
            entry_type: Filtrer par type ('table', 'column', ou None pour tout).

        Returns:
            Liste de dicts triés par score décroissant :
            [{"table": str, "column": str|None, "description": str, "score": float, "type": str}]
        """
        if not self._is_built or self._index is None:
            raise RuntimeError("L'index FAISS n'est pas construit. Appelez build_index() d'abord.")

        if not query or not query.strip():
            return []

        # Générer l'embedding de la requête
        query_embedding = self.embedder.embed(query.strip())
        query_vec       = query_embedding.reshape(1, -1).astype(np.float32)

        # Rechercher dans FAISS (retourne top_k * 3 pour filtrer ensuite)
        search_k = min(top_k * 3, len(self._entries))
        scores, indices = self._index.search(query_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue

            entry = self._entries[idx]

            # Filtrer par type si demandé
            if entry_type and entry.entry_type != entry_type:
                continue

            results.append({
                "table":       entry.table_name,
                "column":      entry.column_name,
                "description": entry.description,
                "score":       float(score),
                "type":        entry.entry_type,
                "tags":        entry.tags,
            })

            if len(results) >= top_k:
                break

        return results

    def search_tables(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Recherche uniquement les tables (pas les colonnes).

        Args:
            query: Requête en langage naturel.
            top_k: Nombre de tables à retourner.

        Returns:
            Liste de tables avec scores.
        """
        return self.search(query, top_k=top_k, entry_type="table")

    def search_columns(self, query: str, table_name: str, top_k: int = 10) -> List[Dict]:
        """
        Recherche les colonnes d'une table spécifique.

        Args:
            query:      Requête en langage naturel.
            table_name: Table dans laquelle chercher.
            top_k:      Nombre de colonnes à retourner.

        Returns:
            Liste de colonnes avec scores.
        """
        all_results = self.search(query, top_k=top_k * 3, entry_type="column")
        return [r for r in all_results if r["table"] == table_name][:top_k]

    def get_top_tables(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retourne les noms des tables les plus pertinentes.

        Args:
            query: Requête en langage naturel.
            top_k: Nombre de tables.

        Returns:
            Liste de noms de tables triés par pertinence.
        """
        results = self.search_tables(query, top_k=top_k)
        return [r["table"] for r in results]

    # ── Persistance ───────────────────────────────────────────────────────────

    def save_index(
        self,
        index_path:    Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> None:
        """
        Sauvegarde l'index FAISS et les métadonnées sur disque.

        Args:
            index_path:    Chemin du fichier index FAISS (.index).
            metadata_path: Chemin du fichier métadonnées (.json).
        """
        import faiss

        if not self._is_built:
            raise RuntimeError("L'index n'est pas construit. Appelez build_index() d'abord.")

        index_path    = index_path    or self.index_path
        metadata_path = metadata_path or self.metadata_path

        # Créer les répertoires si nécessaire
        os.makedirs(os.path.dirname(index_path),    exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        # Sauvegarder l'index FAISS
        faiss.write_index(self._index, index_path)
        logger.info("Index FAISS sauvegardé → %s", index_path)

        # Sauvegarder les métadonnées
        metadata = {
            "entries":    [e.to_dict() for e in self._entries],
            "model_name": self.embedder.model_name,
            "dimension":  self.embedder.dimension,
            "count":      len(self._entries),
            "saved_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("Métadonnées sauvegardées → %s", metadata_path)

    def load_index(
        self,
        index_path:    Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> bool:
        """
        Charge l'index FAISS et les métadonnées depuis le disque.

        Args:
            index_path:    Chemin du fichier index FAISS.
            metadata_path: Chemin du fichier métadonnées.

        Returns:
            True si chargé avec succès, False sinon.
        """
        import faiss

        index_path    = index_path    or self.index_path
        metadata_path = metadata_path or self.metadata_path

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning(
                "Fichiers d'index introuvables — index=%s metadata=%s",
                index_path, metadata_path,
            )
            return False

        try:
            # Charger l'index FAISS
            self._index = faiss.read_index(index_path)
            logger.info("Index FAISS chargé depuis %s (%d vecteurs)", index_path, self._index.ntotal)

            # Charger les métadonnées
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self._entries  = [IndexEntry.from_dict(e) for e in metadata["entries"]]
            self._is_built = True

            logger.info(
                "Index chargé — %d entrées, modèle=%s",
                len(self._entries), metadata.get("model_name", "unknown"),
            )
            return True

        except Exception as exc:
            logger.error("Erreur lors du chargement de l'index : %s", str(exc), exc_info=True)
            return False

    # ── Utilitaires ───────────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        """Indique si l'index est construit et prêt."""
        return self._is_built

    @property
    def entry_count(self) -> int:
        """Nombre d'entrées dans l'index."""
        return len(self._entries)

    def get_stats(self) -> Dict:
        """Retourne les statistiques de l'index."""
        if not self._is_built:
            return {"built": False}

        table_count  = sum(1 for e in self._entries if e.entry_type == "table")
        column_count = sum(1 for e in self._entries if e.entry_type == "column")

        return {
            "built":        True,
            "total_entries": len(self._entries),
            "table_entries": table_count,
            "column_entries": column_count,
            "model_name":   self.embedder.model_name,
            "dimension":    self.embedder.dimension,
        }
