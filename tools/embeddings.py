from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from tools.chunking import ChunkRecord

logger = logging.getLogger(__name__)


def _to_e5_query_style(text: str) -> str:
    # E5 attend généralement des préfixes "query:" / "passage:" pour de meilleures perfs.
    # Ici, on encode des passages (chunks).
    return f"passage: {text}"


def embed_chunks(
    chunks: List[ChunkRecord],
    model_name: str,
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode une liste de chunks en embeddings.
    Retour:
      - embeddings: np.ndarray shape (n, dim) float32
      - chunk_ids: List[str] alignée avec embeddings
    """
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32), []

    logger.info(f"Chargement du modèle d'embedding: {model_name}")
    model = SentenceTransformer(model_name, device="cpu")

    texts = [_to_e5_query_style(c.text) for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    logger.info(f"Encodage embeddings: {len(texts)} chunks (batch_size={batch_size})")
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # utile pour cosine sim avec FAISS IP
    )

    # Sécurise le type pour FAISS ensuite
    vectors = np.asarray(vectors, dtype=np.float32)

    logger.info(f"Embeddings générés: shape={vectors.shape}")
    return vectors, chunk_ids
