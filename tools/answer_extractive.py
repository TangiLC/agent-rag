from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from tools.reranker import RerankedItem

logger = logging.getLogger(__name__)


def build_extractive_answer(
    query: str,
    reranked: Sequence[RerankedItem],
    chunk_store: Dict[str, dict],
    top_n: int = 4,
    max_chars_per_chunk: int = 800,
) -> str:
    """
    Réponse 100% extractive: concatène des extraits issus du corpus.
    - Pas de reformulation
    - Pas de connaissances externes
    - Sources strictes
    """
    kept = list(reranked[:top_n])
    if not kept:
        return "Non trouvé dans le corpus."

    parts: List[str] = []
    sources: List[str] = []

    for item in kept:
        meta = chunk_store[item.chunk_id]
        text = meta["text"].strip()

        # extrait raisonnable (évite d'inonder la console)
        excerpt = text[:max_chars_per_chunk].strip()
        parts.append(excerpt)

        sources.append(
            f"- {meta['doc_id']} p.{meta['page']} | {item.chunk_id} | score={item.score:.4f}"
        )

    answer = []
    answer.append("=== Extraits pertinents ===")
    answer.append("")
    for i, p in enumerate(parts, start=1):
        answer.append(f"[{i}] {p}")
        answer.append("")

    answer.append("=== Sources ===")
    answer.extend(sources)

    return "\n".join(answer)
