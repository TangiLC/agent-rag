from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    doc_id: str
    page: int
    text: str


def load_chunk_store(chunks_jsonl_path: Path) -> Dict[str, dict]:
    store: Dict[str, dict] = {}
    with chunks_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            store[obj["chunk_id"]] = {
                "doc_id": obj["doc_id"],
                "page": int(obj["page"]),
                "text": obj["text"],
            }
    return store


def load_chunk_ids(chunk_ids_path: Path) -> List[str]:
    with chunk_ids_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_embeddings(embeddings_npy_path: Path) -> np.ndarray:
    arr = np.load(str(embeddings_npy_path))
    return np.asarray(arr, dtype=np.float32)


def _to_e5_query_style(text: str) -> str:
    return f"query: {text}"


def embed_query_with_model(model: SentenceTransformer, query: str) -> np.ndarray:
    vec = model.encode(
        [_to_e5_query_style(query)],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vec[0], dtype=np.float32)


def top_k_by_dot(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    chunk_ids: List[str],
    top_k: int,
) -> List[Tuple[str, float]]:
    """
    Retourne [(chunk_id, score)] par dot-product, triés décroissants.
    embeddings doivent être normalisés et query_vec normalisé.
    """
    scores = embeddings @ query_vec
    k = min(top_k, scores.shape[0])
    if k <= 0:
        return []

    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(chunk_ids[int(i)], float(scores[int(i)])) for i in idx]
