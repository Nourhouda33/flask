"""
Script d'initialisation des embeddings — Healthcare AI Platform
À exécuter UNE SEULE FOIS au démarrage ou après modification du schéma.

Ce script :
  1. Charge le schéma MySQL (via SchemaBuilder)
  2. Génère les embeddings de toutes les tables et colonnes
  3. Construit l'index FAISS
  4. Sauvegarde l'index et les métadonnées sur disque

Usage :
    # Depuis le répertoire backend/
    python embeddings/init_embeddings.py

    # Avec options :
    python embeddings/init_embeddings.py --model biolord --device cpu
    python embeddings/init_embeddings.py --model minilm --force-rebuild
"""

import os
import sys
import time
import logging
import argparse

# Ajouter le répertoire backend au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration du logging pour le script
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("init_embeddings")


# ─────────────────────────────────────────────────────────────────────────────
#  Fonctions principales
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies() -> bool:
    """
    Vérifie que toutes les dépendances nécessaires sont installées.

    Returns:
        True si tout est OK, False sinon.
    """
    missing = []

    try:
        import numpy
        logger.info("  ✓ numpy %s", numpy.__version__)
    except ImportError:
        missing.append("numpy")

    try:
        import faiss
        logger.info("  ✓ faiss-cpu")
    except ImportError:
        missing.append("faiss-cpu")

    try:
        import sentence_transformers
        logger.info("  ✓ sentence-transformers %s", sentence_transformers.__version__)
    except ImportError:
        missing.append("sentence-transformers")

    try:
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        logger.info("  ✓ torch %s (%s)", torch.__version__, device)
    except ImportError:
        missing.append("torch")

    if missing:
        logger.error("Dépendances manquantes : %s", ", ".join(missing))
        logger.error("Installez-les avec : pip install %s", " ".join(missing))
        return False

    return True


def build_and_save_index(
    model_key:     str = "biolord",
    index_path:    str = "embeddings/faiss_store/healthcare.index",
    metadata_path: str = "embeddings/faiss_store/metadata.json",
    device:        str = "auto",
    force_rebuild: bool = False,
) -> bool:
    """
    Construit et sauvegarde l'index FAISS.

    Args:
        model_key:     Modèle d'embedding ('biolord', 'biobert', 'minilm').
        index_path:    Chemin de sauvegarde de l'index FAISS.
        metadata_path: Chemin de sauvegarde des métadonnées.
        device:        Device ('cpu', 'cuda', 'auto').
        force_rebuild: Forcer la reconstruction même si l'index existe.

    Returns:
        True si succès, False sinon.
    """
    # Vérifier si l'index existe déjà
    if not force_rebuild and os.path.exists(index_path) and os.path.exists(metadata_path):
        logger.info("Index FAISS déjà existant — utilisez --force-rebuild pour reconstruire")
        logger.info("  Index    : %s", index_path)
        logger.info("  Metadata : %s", metadata_path)
        return True

    total_start = time.perf_counter()

    # ── Étape 1 : Charger le schéma ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ÉTAPE 1/4 — Chargement du schéma de la base de données")
    logger.info("=" * 60)

    from database.schema_builder import SCHEMA, schema_builder

    tables  = schema_builder.get_all_tables()
    columns = sum(len(SCHEMA[t]["columns"]) for t in tables)
    logger.info("  Tables   : %d (%s)", len(tables), ", ".join(tables))
    logger.info("  Colonnes : %d au total", columns)
    logger.info("  Entrées FAISS prévues : %d (tables + colonnes)", len(tables) + columns)

    # ── Étape 2 : Charger le modèle d'embedding ────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("ÉTAPE 2/4 — Chargement du modèle d'embedding : %s", model_key)
    logger.info("=" * 60)

    from embeddings.biomedical_embeddings import BiomedicalEmbedder, MODEL_CONFIGS

    model_info = MODEL_CONFIGS.get(model_key, {})
    logger.info("  Modèle   : %s", model_info.get("name", model_key))
    logger.info("  Type     : %s", model_info.get("type", "unknown"))
    logger.info("  Device   : %s", device)

    embedder = BiomedicalEmbedder(model_key=model_key, device=device)

    # Forcer le chargement du modèle maintenant (pas lazy)
    logger.info("  Chargement en cours...")
    embed_start = time.perf_counter()
    _ = embedder.embed("test médical")  # Force le chargement
    embed_load_time = round((time.perf_counter() - embed_start) * 1000, 1)
    logger.info("  Modèle chargé en %dms — dimension=%d", embed_load_time, embedder.dimension)

    # ── Étape 3 : Construire l'index FAISS ─────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("ÉTAPE 3/4 — Construction de l'index FAISS")
    logger.info("=" * 60)

    from embeddings.faiss_index import FAISSTableIndex

    faiss_index = FAISSTableIndex(
        embedder=embedder,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    build_start = time.perf_counter()
    faiss_index.build_index(schema=SCHEMA)
    build_time = round((time.perf_counter() - build_start) * 1000, 1)

    stats = faiss_index.get_stats()
    logger.info("  Entrées totales  : %d", stats["total_entries"])
    logger.info("  Entrées tables   : %d", stats["table_entries"])
    logger.info("  Entrées colonnes : %d", stats["column_entries"])
    logger.info("  Temps de build   : %dms", build_time)

    # ── Étape 4 : Sauvegarder l'index ─────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("ÉTAPE 4/4 — Sauvegarde de l'index sur disque")
    logger.info("=" * 60)

    faiss_index.save_index(index_path=index_path, metadata_path=metadata_path)

    # Vérifier les fichiers créés
    if os.path.exists(index_path):
        size_kb = os.path.getsize(index_path) / 1024
        logger.info("  ✓ Index FAISS    : %s (%.1f KB)", index_path, size_kb)
    if os.path.exists(metadata_path):
        size_kb = os.path.getsize(metadata_path) / 1024
        logger.info("  ✓ Métadonnées    : %s (%.1f KB)", metadata_path, size_kb)

    # ── Résumé ─────────────────────────────────────────────────────────────
    total_time = round((time.perf_counter() - total_start) * 1000, 1)
    logger.info("")
    logger.info("=" * 60)
    logger.info("INITIALISATION TERMINÉE avec succès")
    logger.info("=" * 60)
    logger.info("  Modèle       : %s", embedder.model_name)
    logger.info("  Dimension    : %d", embedder.dimension)
    logger.info("  Entrées      : %d", stats["total_entries"])
    logger.info("  Temps total  : %dms", total_time)
    logger.info("  Cache stats  : %s", embedder.get_cache_stats())

    return True


def verify_index(
    index_path:    str = "embeddings/faiss_store/healthcare.index",
    metadata_path: str = "embeddings/faiss_store/metadata.json",
    model_key:     str = "biolord",
) -> bool:
    """
    Vérifie que l'index FAISS est fonctionnel en effectuant des recherches de test.

    Args:
        index_path:    Chemin de l'index FAISS.
        metadata_path: Chemin des métadonnées.
        model_key:     Modèle d'embedding.

    Returns:
        True si l'index est fonctionnel, False sinon.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("VÉRIFICATION DE L'INDEX")
    logger.info("=" * 60)

    from embeddings.biomedical_embeddings import BiomedicalEmbedder
    from embeddings.faiss_index import FAISSTableIndex

    embedder    = BiomedicalEmbedder(model_key=model_key)
    faiss_index = FAISSTableIndex(
        embedder=embedder,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    if not faiss_index.load_index():
        logger.error("Impossible de charger l'index FAISS")
        return False

    # Requêtes de test
    test_queries = [
        ("patients diabétiques", ["Patient", "Medical_records"]),
        ("médecins cardiologues", ["Medical_staff", "Service"]),
        ("consultations diagnostics", ["Consultation"]),
        ("groupe sanguin allergies", ["Medical_records"]),
        ("service hospitalier département", ["Service"]),
    ]

    all_passed = True
    for query, expected_tables in test_queries:
        results = faiss_index.search_tables(query, top_k=3)
        found_tables = [r["table"] for r in results]
        top_score    = results[0]["score"] if results else 0

        # Vérifier qu'au moins une table attendue est dans les résultats
        match = any(t in found_tables for t in expected_tables)
        status = "✓" if match else "✗"

        logger.info(
            "  %s Query: %-40s → %s (score=%.3f)",
            status, f'"{query}"', found_tables[:2], top_score,
        )

        if not match:
            all_passed = False

    if all_passed:
        logger.info("")
        logger.info("  ✓ Toutes les vérifications ont réussi")
    else:
        logger.warning("")
        logger.warning("  ✗ Certaines vérifications ont échoué")

    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Initialise les embeddings biomédicaux et l'index FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python embeddings/init_embeddings.py
  python embeddings/init_embeddings.py --model minilm --device cpu
  python embeddings/init_embeddings.py --force-rebuild --verify
        """,
    )
    parser.add_argument(
        "--model",
        choices=["biolord", "biobert", "minilm"],
        default="biolord",
        help="Modèle d'embedding à utiliser (défaut: biolord)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device de calcul (défaut: auto)",
    )
    parser.add_argument(
        "--index-path",
        default="embeddings/faiss_store/healthcare.index",
        help="Chemin de l'index FAISS",
    )
    parser.add_argument(
        "--metadata-path",
        default="embeddings/faiss_store/metadata.json",
        help="Chemin des métadonnées",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Forcer la reconstruction même si l'index existe",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Vérifier l'index après construction",
    )
    parser.add_argument(
        "--check-deps-only",
        action="store_true",
        help="Vérifier uniquement les dépendances",
    )
    return parser.parse_args()


def main() -> int:
    """
    Point d'entrée principal du script.

    Returns:
        Code de sortie (0 = succès, 1 = erreur).
    """
    args = parse_args()

    logger.info("Healthcare AI Platform — Initialisation des Embeddings")
    logger.info("=" * 60)

    # ── Vérification des dépendances ───────────────────────────────────────
    logger.info("Vérification des dépendances...")
    if not check_dependencies():
        return 1

    if args.check_deps_only:
        logger.info("Toutes les dépendances sont disponibles.")
        return 0

    # ── Construction de l'index ────────────────────────────────────────────
    success = build_and_save_index(
        model_key=args.model,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        device=args.device,
        force_rebuild=args.force_rebuild,
    )

    if not success:
        logger.error("Échec de la construction de l'index")
        return 1

    # ── Vérification optionnelle ───────────────────────────────────────────
    if args.verify:
        verified = verify_index(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            model_key=args.model,
        )
        if not verified:
            logger.warning("La vérification a détecté des problèmes")
            return 1

    logger.info("")
    logger.info("L'index FAISS est prêt. Vous pouvez démarrer l'application.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
