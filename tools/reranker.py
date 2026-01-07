from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Optional

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


def _get_best_device() -> str:
    """
    Priorité:
      1) cuda (NVIDIA)
      2) mps  (Apple Silicon)
      3) cpu
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class RerankedItem:
    chunk_id: str
    score: float


def load_reranker_model(model_name: str, device: Optional[str] = None) -> CrossEncoder:
    """
    Charge le cross-encoder une seule fois.
    Utilise le GPU local si disponible, sinon CPU.
    """
    if device is None:
        device = _get_best_device()

    logger.info(f"Chargement reranker: {model_name} (device={device})")
    return CrossEncoder(model_name, device=device)


def rerank_with_model(
    query: str,
    chunk_texts: Sequence[str],
    chunk_ids: Sequence[str],
    model: CrossEncoder,
) -> List[RerankedItem]:
    """
    Score cross-encoder (query, chunk) puis tri décroissant.
    """
    if len(chunk_texts) != len(chunk_ids):
        raise ValueError("chunk_texts et chunk_ids doivent être alignés")
    if not chunk_texts:
        return []

    pairs = [(query, t) for t in chunk_texts]
    scores = model.predict(pairs)

    items = [
        RerankedItem(chunk_id=cid, score=float(s)) for cid, s in zip(chunk_ids, scores)
    ]
    items.sort(key=lambda x: x.score, reverse=True)
    return items
