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
    """
    Charge chunks.jsonl en mémoire (mapping chunk_id -> {doc_id, page, text}).
    """
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


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    vec = model.encode(
        [_to_e5_query_style(query)],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vec[0], dtype=np.float32)


def brute_force_search(
    query: str,
    embeddings: np.ndarray,
    chunk_ids: List[str],
    chunk_store: Dict[str, dict],
    model_name: str,
    top_k: int = 5,
) -> List[RetrievedChunk]:
    """
    Recherche brute-force: cosine(query, chunk) via dot product sur embeddings normalisés.
    """
    if embeddings.size == 0 or not chunk_ids:
        return []

    if embeddings.shape[0] != len(chunk_ids):
        raise ValueError(
            f"Mismatch: embeddings={embeddings.shape[0]} vs chunk_ids={len(chunk_ids)}"
        )

    logger.info(f"Chargement du modèle (requête): {model_name}")
    model = SentenceTransformer(model_name, device="cpu")

    logger.info(f"Embedding requête (len={len(query)})")
    q = embed_query(model, query)  # shape (dim,)

    logger.info(f"Brute-force similarity sur {embeddings.shape[0]} vecteurs")
    scores = embeddings @ q  # (N,) dot product

    k = min(top_k, scores.shape[0])
    if k <= 0:
        return []

    # top-k sans trier tout le tableau
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    out: List[RetrievedChunk] = []
    for i in idx:
        cid = chunk_ids[int(i)]
        meta = chunk_store.get(cid)
        if not meta:
            # si jamais le store n'a pas le chunk (artefacts incohérents)
            continue
        out.append(
            RetrievedChunk(
                chunk_id=cid,
                score=float(scores[int(i)]),
                doc_id=meta["doc_id"],
                page=int(meta["page"]),
                text=meta["text"],
            )
        )
    return out


def top_k_by_dot(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    chunk_ids: List[str],
    top_k: int,
) -> List[Tuple[str, float]]:
    """
    Retourne [(chunk_id, score)] par dot-product, triés décroissants.
    embeddings doivent être normalisés, query_vec normalisé.
    """
    scores = embeddings @ query_vec
    k = min(top_k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(chunk_ids[int(i)], float(scores[int(i)])) for i in idx]
