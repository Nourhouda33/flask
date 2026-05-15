"""
Package embeddings — BioBERT, BioLORD, FAISS index pour le Table Matching.
"""

from embeddings.biomedical_embeddings import BiomedicalEmbedder, EmbeddingCache, create_embedder
from embeddings.faiss_index import FAISSTableIndex, IndexEntry

__all__ = [
    "BiomedicalEmbedder",
    "EmbeddingCache",
    "create_embedder",
    "FAISSTableIndex",
    "IndexEntry",
]
