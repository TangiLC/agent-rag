import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import config
from tools.pdf_loader import iter_pdf_paths, load_corpus_pages
from tools.chunking import chunk_pages
from tools.embeddings import embed_chunks
from tools.vector_store_faiss import build_faiss_index, save_faiss_index


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:INFO|%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _list_corpus_facts(pdf_dir: Path) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for p in iter_pdf_paths(pdf_dir):
        st = p.stat()
        facts.append(
            {
                "name": p.name,
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
            }
        )
    return facts


def _write_chunks_jsonl(chunks, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            obj = {
                "chunk_id": ch.chunk_id,
                "doc_id": ch.doc_id,
                "page": int(ch.page),
                "text": ch.text,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_chunk_ids(chunk_ids: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cid in chunk_ids:
            f.write(cid + "\n")


def _write_manifest(manifest: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("BUILD – démarrage")

    # 0) Snapshot corpus
    corpus_facts = _list_corpus_facts(config.PDF_DIR)
    logger.info(f"Corpus: {len(corpus_facts)} PDFs trouvés dans {config.PDF_DIR}")
    if not corpus_facts:
        logger.info("Aucun PDF trouvé. Arrêt.")
        return

    # 1) Load pages
    logger.info("Chargement des pages PDF (PyMuPDF)")
    pages = load_corpus_pages(
        pdf_dir=config.PDF_DIR,
        max_pages_per_pdf=config.MAX_PAGES_PER_PDF,
        min_chars_per_page=config.MIN_CHARS_PER_PAGE,
    )
    logger.info(f"{len(pages)} pages chargées")
    if not pages:
        logger.info("Aucune page exploitable. Arrêt.")
        return

    # 2) Chunking
    logger.info(
        "Chunking "
        f"(taille={config.CHUNK_SIZE_CHARS}, overlap={config.CHUNK_OVERLAP_CHARS}, "
        f"fenêtre=[{config.CHUNK_SOFT_MIN},{config.CHUNK_SOFT_MAX}], min={config.MIN_CHUNK_CHARS})"
    )
    chunks = chunk_pages(
        pages=pages,
        chunk_size_chars=config.CHUNK_SIZE_CHARS,
        chunk_overlap_chars=config.CHUNK_OVERLAP_CHARS,
        min_chunk_chars=config.MIN_CHUNK_CHARS,
        soft_min=config.CHUNK_SOFT_MIN,
        soft_max=config.CHUNK_SOFT_MAX,
    )
    logger.info(f"{len(chunks)} chunks générés")
    if not chunks:
        logger.info("Aucun chunk généré. Arrêt.")
        return

    # 3) Embeddings
    logger.info(f"Embeddings ({config.EMBEDDING_MODEL_NAME})")
    vectors, chunk_ids = embed_chunks(
        chunks=chunks,
        model_name=config.EMBEDDING_MODEL_NAME,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )
    if vectors.size == 0 or not chunk_ids:
        logger.info("Aucun embedding généré. Arrêt.")
        return

    if vectors.shape[0] != len(chunk_ids):
        logger.info(
            f"Incohérence embeddings: {vectors.shape[0]} vecteurs vs {len(chunk_ids)} ids"
        )
        logger.info("Arrêt.")
        return

    # 4) Écriture artefacts “texte + embeddings”
    logger.info(f"Écriture artefacts dans {config.ARTIFACTS_DIR}")
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    _write_chunks_jsonl(chunks, config.CHUNKS_JSONL_PATH)
    _write_chunk_ids(chunk_ids, config.CHUNK_IDS_PATH)
    np.save(str(config.EMBEDDINGS_NPY_PATH), vectors)

    logger.info(f"OK embeddings: shape={vectors.shape}")

    # 5) FAISS
    logger.info("Construction FAISS")
    faiss_index = build_faiss_index(
        vectors, use_ip=getattr(config, "FAISS_USE_IP", True)
    )
    save_faiss_index(faiss_index, config.FAISS_INDEX_PATH)

    # 6) Manifest
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf_dir": str(config.PDF_DIR),
        "corpus_facts": corpus_facts,
        "chunking": {
            "chunk_size_chars": config.CHUNK_SIZE_CHARS,
            "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
            "min_chunk_chars": config.MIN_CHUNK_CHARS,
            "soft_min": config.CHUNK_SOFT_MIN,
            "soft_max": config.CHUNK_SOFT_MAX,
        },
        "embedding": {
            "model_name": config.EMBEDDING_MODEL_NAME,
            "batch_size": config.EMBEDDING_BATCH_SIZE,
            "dim": int(vectors.shape[1]),
            "normalized": True,
        },
        "faiss": {
            "use_ip": bool(getattr(config, "FAISS_USE_IP", True)),
            "ntotal": int(faiss_index.ntotal),
        },
        "artefacts": {
            "chunks_jsonl": str(config.CHUNKS_JSONL_PATH),
            "chunk_ids": str(config.CHUNK_IDS_PATH),
            "embeddings_npy": str(config.EMBEDDINGS_NPY_PATH),
            "faiss_index": str(config.FAISS_INDEX_PATH),
        },
    }
    _write_manifest(manifest, config.MANIFEST_PATH)

    logger.info("BUILD – terminé")


if __name__ == "__main__":
    main()
