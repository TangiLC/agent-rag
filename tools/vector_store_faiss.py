from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_faiss_index(embeddings: np.ndarray, use_ip: bool = True) -> faiss.Index:
    """
    Construit un index FAISS simple.
    - use_ip=True : IndexFlatIP (inner product) recommandé si embeddings normalisés.
    - use_ip=False: IndexFlatL2 (L2 distance).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings doit être un array 2D (N, dim)")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)

    n, dim = embeddings.shape
    if n == 0:
        raise ValueError("embeddings vide")

    index = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
    index.add(embeddings)

    logger.info(
        f"FAISS index construit: N={index.ntotal}, dim={dim}, metric={'IP' if use_ip else 'L2'}"
    )
    return index


def save_faiss_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info(f"FAISS index sauvegardé: {path}")


def load_faiss_index(path: Path) -> faiss.Index:
    index = faiss.read_index(str(path))
    logger.info(f"FAISS index chargé: {path} (N={index.ntotal})")
    return index


def search_faiss(
    index: faiss.Index,
    query_vec: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retourne (scores, idx):
    - scores: (1, k 'float32')
    - idx: (1, k) int64 indices dans l'ordre chunk_ids/embeddings
    """
    if query_vec.ndim != 1:
        raise ValueError("query_vec doit être 1D (dim,)")
    q = query_vec.astype(np.float32, copy=False).reshape(1, -1)

    scores, idx = index.search(q, top_k)
    return scores, idx
