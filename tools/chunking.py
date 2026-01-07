from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Iterable, List

from tools.pdf_loader import PageRecord


# Le logging est configuré dans le main (orchestrateur).
# Ici, on récupère juste un logger nommé.
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRecord:
    doc_id: str
    page: int
    chunk_id: str
    text: str


_SPLIT_REGEX = re.compile(r"[\.?!\n]")


def chunk_pages(
    pages: Iterable[PageRecord],
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    min_chunk_chars: int,
    soft_min: int,
    soft_max: int,
) -> List[ChunkRecord]:
    """
    Découpe les pages en chunks avec overlap.
    Coupe préférentiellement sur ponctuation forte dans [soft_min, soft_max].
    """
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars doit être < chunk_size_chars")
    if soft_min <= 0 or soft_max <= soft_min:
        raise ValueError("soft_min / soft_max invalides")

    pages_list = list(pages)
    total_pages = len(pages_list)

    logger.info(
        "Début chunking | pages=%d chunk_size=%d overlap=%d soft=[%d,%d] min_chunk=%d",
        total_pages,
        chunk_size_chars,
        chunk_overlap_chars,
        soft_min,
        soft_max,
        min_chunk_chars,
    )

    all_chunks: List[ChunkRecord] = []
    total_chunks = 0

    for src_idx, page in enumerate(pages_list, start=1):
        page_id = f"{page.doc_id}:p{page.page}"
        page_text_len = len((page.text or "").strip())

        logger.info(
            "Traitement page %d/%d | %s | chars=%d",
            src_idx,
            total_pages,
            page_id,
            page_text_len,
        )

        page_chunks = _chunk_single_page(
            page=page,
            chunk_size=chunk_size_chars,
            overlap=chunk_overlap_chars,
            min_chunk_chars=min_chunk_chars,
            soft_min=soft_min,
            soft_max=soft_max,
            src_idx=src_idx,
            total_pages=total_pages,
        )

        all_chunks.extend(page_chunks)
        total_chunks += len(page_chunks)

        logger.info(
            "Fin page %d/%d | %s | chunks=%d | total_chunks=%d",
            src_idx,
            total_pages,
            page_id,
            len(page_chunks),
            total_chunks,
        )

    logger.info(
        "Chunking terminé | pages=%d total_chunks=%d", total_pages, total_chunks
    )
    return all_chunks


def _chunk_single_page(
    page: PageRecord,
    chunk_size: int,
    overlap: int,
    min_chunk_chars: int,
    soft_min: int,
    soft_max: int,
    src_idx: int,
    total_pages: int,
) -> List[ChunkRecord]:
    text = (page.text or "").strip()
    if not text:
        logger.info(
            "Page vide ignorée | page %d/%d | doc=%s page=%d",
            src_idx,
            total_pages,
            page.doc_id,
            page.page,
        )
        return []

    out: List[ChunkRecord] = []
    n = len(text)
    start = 0
    chunk_index = 0

    # Estimation "optimiste" du nombre de chunks (utile pour un n/nn approximatif)
    est_total = (
        max(1, (n + (chunk_size - overlap) - 1) // (chunk_size - overlap))
        if chunk_size > overlap
        else 1
    )

    while start < n:
        hard_end = min(start + chunk_size, n)

        window_start = min(start + soft_min, n)
        window_end = min(start + soft_max, n)

        cut = _find_split(text, window_start, window_end)
        end = cut if cut is not None else hard_end

        chunk_text = text[start:end].strip()
        if len(chunk_text) < min_chunk_chars:
            logger.info(
                "Arrêt page (chunk trop court) | page %d/%d | doc=%s page=%d | "
                "chunk=%d | len=%d < min=%d",
                src_idx,
                total_pages,
                page.doc_id,
                page.page,
                chunk_index + 1,
                len(chunk_text),
                min_chunk_chars,
            )
            break

        chunk_id = f"{page.doc_id}:p{page.page}:c{chunk_index}"

        out.append(
            ChunkRecord(
                doc_id=page.doc_id,
                page=page.page,
                chunk_id=chunk_id,
                text=chunk_text,
            )
        )

        logger.info(
            "Chunk %d/%d | page %d/%d | %s | chars=%d | span=[%d,%d) | split=%s",
            chunk_index + 1,
            est_total,
            src_idx,
            total_pages,
            chunk_id,
            len(chunk_text),
            start,
            end,
            "punct" if cut is not None else "hard",
        )

        chunk_index += 1
        start = max(start + 1, end - overlap)  # +1 pour éviter les boucles infinies

    return out


def _find_split(text: str, start: int, end: int) -> int | None:
    slice_ = text[start:end]
    matches = list(_SPLIT_REGEX.finditer(slice_))
    if not matches:
        return None
    return start + matches[-1].end()
