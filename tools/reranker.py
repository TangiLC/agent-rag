from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankedItem:
    chunk_id: str
    score: float


def rerank(
    query: str,
    chunk_texts: Sequence[str],
    chunk_ids: Sequence[str],
    model_name: str,
) -> List[RerankedItem]:
    """
    Rerank cross-encoder : score(query, chunk_text).
    Retourne une liste triée décroissante par score.
    """
    if len(chunk_texts) != len(chunk_ids):
        raise ValueError("chunk_texts et chunk_ids doivent être alignés")
    if not chunk_texts:
        return []

    logger.info(f"Chargement reranker: {model_name}")
    model = CrossEncoder(model_name, device="cpu")

    pairs = [(query, t) for t in chunk_texts]
    scores = model.predict(pairs)  # numpy array-like

    items = [
        RerankedItem(chunk_id=cid, score=float(s)) for cid, s in zip(chunk_ids, scores)
    ]
    items.sort(key=lambda x: x.score, reverse=True)
    return items
