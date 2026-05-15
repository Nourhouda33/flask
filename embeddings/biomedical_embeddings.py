"""
Embeddings biomédicaux — Healthcare AI Platform
Support multi-modèles avec chargement lazy et cache LRU.

Hiérarchie des modèles (du plus spécialisé au plus générique) :
  1. BioLORD-2023   (FremyCompany/BioLORD-2023)          — meilleur pour NLP médical
  2. BioBERT        (dmis-lab/biobert-base-cased-v1.2)   — robuste domaine biomédical
  3. MiniLM         (sentence-transformers/all-MiniLM-L6-v2) — fallback léger

Architecture :
  BiomedicalEmbedder  → classe principale (lazy loading + LRU cache)
  ModelLoader         → chargement et gestion des modèles HuggingFace
  EmbeddingCache      → cache LRU thread-safe
"""

import os
import hashlib
import logging
import threading
from functools import lru_cache
from typing import List, Optional, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Modèles supportés avec leurs dimensions d'embedding
MODEL_CONFIGS: Dict[str, Dict] = {
    "biolord": {
        "name":      "FremyCompany/BioLORD-2023",
        "dimension": 768,
        "type":      "sentence_transformer",
        "description": "Modèle biomédical optimisé pour la similarité sémantique médicale",
    },
    "biobert": {
        "name":      "dmis-lab/biobert-base-cased-v1.2",
        "dimension": 768,
        "type":      "transformers",
        "description": "BioBERT pré-entraîné sur PubMed et PMC",
    },
    "minilm": {
        "name":      "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "type":      "sentence_transformer",
        "description": "Modèle léger généraliste — fallback",
    },
}

# Ordre de priorité pour le chargement automatique
MODEL_PRIORITY = ["biolord", "biobert", "minilm"]

# Taille maximale du cache LRU (nombre d'embeddings)
DEFAULT_CACHE_SIZE = 1024


# ─────────────────────────────────────────────────────────────────────────────
#  EmbeddingCache — cache LRU thread-safe
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Cache LRU thread-safe pour les embeddings.
    Évite de recalculer les embeddings pour les mêmes textes.
    La clé est le hash MD5 du texte + nom du modèle.
    """

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
        """
        Args:
            max_size: Nombre maximum d'entrées dans le cache.
        """
        self._cache: Dict[str, np.ndarray] = {}
        self._order: List[str]             = []   # Pour LRU eviction
        self._max_size = max_size
        self._lock     = threading.Lock()
        self._hits     = 0
        self._misses   = 0

    @staticmethod
    def _make_key(text: str, model_name: str) -> str:
        """Génère une clé de cache unique pour un texte + modèle."""
        content = f"{model_name}::{text}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Récupère un embedding depuis le cache.

        Args:
            text:       Texte dont on cherche l'embedding.
            model_name: Nom du modèle utilisé.

        Returns:
            np.ndarray si trouvé, None sinon.
        """
        key = self._make_key(text, model_name)
        with self._lock:
            if key in self._cache:
                # Déplacer en fin de liste (LRU — most recently used)
                self._order.remove(key)
                self._order.append(key)
                self._hits += 1
                return self._cache[key].copy()
            self._misses += 1
            return None

    def put(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """
        Stocke un embedding dans le cache.

        Args:
            text:       Texte source.
            model_name: Nom du modèle.
            embedding:  Vecteur d'embedding à stocker.
        """
        key = self._make_key(text, model_name)
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self._max_size:
                # Évincer le moins récemment utilisé
                oldest_key = self._order.pop(0)
                del self._cache[oldest_key]

            self._cache[key] = embedding.copy()
            self._order.append(key)

    def clear(self) -> None:
        """Vide le cache."""
        with self._lock:
            self._cache.clear()
            self._order.clear()
            self._hits   = 0
            self._misses = 0

    @property
    def stats(self) -> Dict:
        """Retourne les statistiques du cache."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size":      len(self._cache),
                "max_size":  self._max_size,
                "hits":      self._hits,
                "misses":    self._misses,
                "hit_rate":  round(self._hits / total, 3) if total > 0 else 0.0,
            }


# ─────────────────────────────────────────────────────────────────────────────
#  BiomedicalEmbedder — classe principale
# ─────────────────────────────────────────────────────────────────────────────

class BiomedicalEmbedder:
    """
    Générateur d'embeddings biomédicaux avec chargement lazy et cache LRU.

    Supporte BioBERT, BioLORD et MiniLM (fallback).
    Le modèle est chargé à la première utilisation (lazy loading).

    Usage:
        embedder = BiomedicalEmbedder(model_key="biolord")
        vec = embedder.embed("patient diabétique avec hypertension")
        vecs = embedder.embed_batch(["texte1", "texte2"])
    """

    def __init__(
        self,
        model_key:  str = "biolord",
        cache_size: int = DEFAULT_CACHE_SIZE,
        device:     str = "auto",
    ):
        """
        Args:
            model_key:  Clé du modèle ('biolord', 'biobert', 'minilm').
            cache_size: Taille du cache LRU.
            device:     'cpu', 'cuda', ou 'auto' (détection automatique).
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(
                f"Modèle inconnu : {model_key!r}. "
                f"Valeurs acceptées : {list(MODEL_CONFIGS.keys())}"
            )

        self.model_key    = model_key
        self.model_config = MODEL_CONFIGS[model_key]
        self.model_name   = self.model_config["name"]
        self.dimension    = self.model_config["dimension"]
        self.model_type   = self.model_config["type"]

        # Détection du device
        self.device = self._resolve_device(device)

        # Lazy loading — modèle non chargé à l'init
        self._model      = None
        self._tokenizer  = None
        self._load_lock  = threading.Lock()
        self._is_loaded  = False

        # Cache LRU
        self._cache = EmbeddingCache(max_size=cache_size)

        logger.info(
            "BiomedicalEmbedder initialisé — model=%s device=%s (lazy)",
            self.model_name, self.device,
        )

    # ── Chargement du modèle ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Résout le device à utiliser (auto-détection CUDA)."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def _load_model(self) -> None:
        """
        Charge le modèle en mémoire (thread-safe, appelé une seule fois).
        Utilise sentence-transformers pour BioLORD et MiniLM,
        et HuggingFace transformers pour BioBERT.
        """
        with self._load_lock:
            if self._is_loaded:
                return  # Double-check locking

            logger.info("Chargement du modèle %s sur %s...", self.model_name, self.device)

            try:
                if self.model_type == "sentence_transformer":
                    self._load_sentence_transformer()
                else:
                    self._load_transformers_model()

                self._is_loaded = True
                logger.info("Modèle %s chargé avec succès", self.model_name)

            except Exception as exc:
                logger.error(
                    "Échec du chargement de %s : %s — tentative fallback MiniLM",
                    self.model_name, str(exc),
                )
                self._load_fallback()

    def _load_sentence_transformer(self) -> None:
        """Charge un modèle sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)
        # Mettre à jour la dimension réelle
        self.dimension = self._model.get_sentence_embedding_dimension()

    def _load_transformers_model(self) -> None:
        """Charge un modèle HuggingFace transformers (BioBERT)."""
        from transformers import AutoTokenizer, AutoModel
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model     = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def _load_fallback(self) -> None:
        """Charge le modèle MiniLM comme fallback de dernier recours."""
        fallback_name = MODEL_CONFIGS["minilm"]["name"]
        logger.warning("Chargement du fallback : %s", fallback_name)
        from sentence_transformers import SentenceTransformer
        self._model     = SentenceTransformer(fallback_name, device=self.device)
        self.model_name = fallback_name
        self.model_key  = "minilm"
        self.model_type = "sentence_transformer"
        self.dimension  = self._model.get_sentence_embedding_dimension()
        self._is_loaded = True

    def _ensure_loaded(self) -> None:
        """S'assure que le modèle est chargé avant utilisation."""
        if not self._is_loaded:
            self._load_model()

    # ── Méthodes d'embedding ──────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """
        Génère l'embedding d'un texte unique.
        Utilise le cache LRU pour éviter les recalculs.

        Args:
            text: Texte à encoder (phrase, description, requête médicale).

        Returns:
            Vecteur numpy de forme (dimension,) normalisé L2.
        """
        if not text or not text.strip():
            logger.warning("Texte vide — retour d'un vecteur zéro")
            return np.zeros(self.dimension, dtype=np.float32)

        text = text.strip()

        # Vérifier le cache
        cached = self._cache.get(text, self.model_name)
        if cached is not None:
            return cached

        # Calculer l'embedding
        self._ensure_loaded()
        embedding = self._compute_embedding(text)

        # Normaliser L2 pour la similarité cosinus
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Stocker dans le cache
        self._cache.put(text, self.model_name, embedding)

        return embedding

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Génère les embeddings d'une liste de textes.
        Optimisé par batch pour réduire les appels GPU/CPU.
        Utilise le cache pour les textes déjà calculés.

        Args:
            texts:         Liste de textes à encoder.
            batch_size:    Taille des batches pour le traitement.
            show_progress: Afficher une barre de progression.

        Returns:
            Matrice numpy de forme (len(texts), dimension) normalisée L2.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        self._ensure_loaded()

        # Séparer les textes en cache / à calculer
        results    = [None] * len(texts)
        to_compute = []   # (index_original, texte)

        for i, text in enumerate(texts):
            text = (text or "").strip()
            if not text:
                results[i] = np.zeros(self.dimension, dtype=np.float32)
                continue
            cached = self._cache.get(text, self.model_name)
            if cached is not None:
                results[i] = cached
            else:
                to_compute.append((i, text))

        # Calculer les embeddings manquants par batch
        if to_compute:
            indices = [item[0] for item in to_compute]
            texts_to_embed = [item[1] for item in to_compute]

            computed = self._compute_batch(texts_to_embed, batch_size, show_progress)

            for idx, (orig_idx, text) in enumerate(to_compute):
                emb  = computed[idx]
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                self._cache.put(text, self.model_name, emb)
                results[orig_idx] = emb

        return np.vstack(results).astype(np.float32)

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Calcule l'embedding d'un texte unique selon le type de modèle."""
        if self.model_type == "sentence_transformer":
            return self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ).astype(np.float32)
        else:
            return self._compute_transformers_embedding(text)

    def _compute_batch(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Calcule les embeddings d'un batch de textes."""
        if self.model_type == "sentence_transformer":
            return self._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=show_progress,
            ).astype(np.float32)
        else:
            # Traitement par batch pour les modèles transformers
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                for text in batch:
                    all_embeddings.append(self._compute_transformers_embedding(text))
            return np.vstack(all_embeddings).astype(np.float32)

    def _compute_transformers_embedding(self, text: str) -> np.ndarray:
        """
        Calcule l'embedding via HuggingFace transformers (BioBERT).
        Utilise le mean pooling sur les hidden states.
        """
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean pooling sur les tokens (en ignorant le padding)
        token_embeddings  = outputs.last_hidden_state
        attention_mask    = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings    = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask          = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled       = (sum_embeddings / sum_mask).squeeze(0)

        return mean_pooled.cpu().numpy().astype(np.float32)

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Calcule la similarité cosinus entre deux textes.

        Args:
            text_a: Premier texte.
            text_b: Deuxième texte.

        Returns:
            Score de similarité entre -1 et 1 (1 = identiques).
        """
        emb_a = self.embed(text_a)
        emb_b = self.embed(text_b)
        return float(np.dot(emb_a, emb_b))

    def get_cache_stats(self) -> Dict:
        """Retourne les statistiques du cache."""
        return self._cache.stats

    def clear_cache(self) -> None:
        """Vide le cache d'embeddings."""
        self._cache.clear()
        logger.info("Cache d'embeddings vidé")

    @property
    def is_loaded(self) -> bool:
        """Indique si le modèle est chargé en mémoire."""
        return self._is_loaded

    def get_info(self) -> Dict:
        """Retourne les informations sur le modèle actif."""
        return {
            "model_key":   self.model_key,
            "model_name":  self.model_name,
            "dimension":   self.dimension,
            "device":      self.device,
            "is_loaded":   self._is_loaded,
            "cache_stats": self._cache.stats,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Factory — création depuis la config
# ─────────────────────────────────────────────────────────────────────────────

def create_embedder(
    model_key:  Optional[str] = None,
    cache_size: int = DEFAULT_CACHE_SIZE,
    device:     str = "auto",
) -> BiomedicalEmbedder:
    """
    Crée un BiomedicalEmbedder en essayant les modèles par ordre de priorité.
    Si le modèle demandé n'est pas disponible, tente le suivant.

    Args:
        model_key:  Clé du modèle ('biolord', 'biobert', 'minilm').
                    Si None, utilise la variable d'env BIOBERT_MODEL ou biolord.
        cache_size: Taille du cache LRU.
        device:     Device ('cpu', 'cuda', 'auto').

    Returns:
        BiomedicalEmbedder configuré.
    """
    if model_key is None:
        # Détecter depuis les variables d'env
        env_model = os.getenv("BIOLORD_MODEL", "")
        if "BioLORD" in env_model or "biolord" in env_model.lower():
            model_key = "biolord"
        elif os.getenv("BIOBERT_MODEL", ""):
            model_key = "biobert"
        else:
            model_key = "biolord"  # Défaut

    return BiomedicalEmbedder(
        model_key=model_key,
        cache_size=cache_size,
        device=device,
    )
