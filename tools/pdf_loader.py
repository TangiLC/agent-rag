from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageRecord:
    doc_id: str
    page: int          # 1-based
    text: str
    source_path: str


def iter_pdf_paths(pdf_dir: Path) -> Iterable[Path]:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {pdf_dir.resolve()}")
    yield from sorted(pdf_dir.glob("*.pdf"))


def load_pdf_pages(
    pdf_path: Path,
    max_pages: Optional[int] = None,
    min_chars_per_page: int = 0,
) -> list[PageRecord]:
    """
    Charge un PDF et retourne une liste de PageRecord (1 record = 1 page).
    PyMuPDF est robuste sur beaucoup de PDFs web.
    """
    doc_id = pdf_path.name
    records: list[PageRecord] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        # PDF illisible / stream corrompu
        return records

    total = doc.page_count
    limit = total if max_pages is None else min(total, max_pages)

    for idx in range(limit):
        try:
            page = doc.load_page(idx)
            text = page.get_text("text") or ""
            text = _normalize_text(text)
        except Exception:
            continue

        if len(text) < min_chars_per_page:
            continue

        records.append(
            PageRecord(
                doc_id=doc_id,
                page=idx + 1,
                text=text,
                source_path=str(pdf_path.resolve()),
            )
        )

    doc.close()
    return records


def load_corpus_pages(
    pdf_dir: Path,
    max_pages_per_pdf: Optional[int] = None,
    min_chars_per_page: int = 0,
) -> list[PageRecord]:
    all_records: list[PageRecord] = []
    for pdf_path in iter_pdf_paths(pdf_dir):
        all_records.extend(
            load_pdf_pages(
                pdf_path,
                max_pages=max_pages_per_pdf,
                min_chars_per_page=min_chars_per_page,
            )
        )
    return all_records


def _normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s.strip()
