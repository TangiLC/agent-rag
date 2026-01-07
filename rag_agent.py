import json
import logging

import numpy as np

import config
from tools.retrieval_bruteforce import (
    brute_force_search,
    load_chunk_ids,
    load_chunk_store,
    load_embeddings,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:INFO|%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("RUNTIME – démarrage")

    # checks simples
    if not config.MANIFEST_PATH.exists():
        logger.info("Aucun manifest trouvé. Lance: python build_index.py")
        return

    if not (
        config.CHUNKS_JSONL_PATH.exists()
        and config.CHUNK_IDS_PATH.exists()
        and config.EMBEDDINGS_NPY_PATH.exists()
    ):
        logger.info("Artefacts manquants. Lance: python build_index.py")
        return

    # charge artefacts
    logger.info("Chargement artefacts")
    chunk_store = load_chunk_store(config.CHUNKS_JSONL_PATH)
    chunk_ids = load_chunk_ids(config.CHUNK_IDS_PATH)
    embeddings = load_embeddings(config.EMBEDDINGS_NPY_PATH)

    logger.info(f"OK artefacts: chunks={len(chunk_ids)}, embeddings={embeddings.shape}")

    if len(chunk_ids) != embeddings.shape[0]:
        logger.info(
            "Incohérence artefacts (ids vs embeddings). Lance: python build_index.py"
        )
        return

    # boucle interactive simple
    logger.info("Prêt. Tape une question (vide pour quitter).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            break

        results = brute_force_search(
            query=q,
            embeddings=embeddings,
            chunk_ids=chunk_ids,
            chunk_store=chunk_store,
            model_name=config.EMBEDDING_MODEL_NAME,
            top_k=5,
        )

        if not results:
            logger.info("Aucun résultat.")
            continue

        logger.info("Top résultats (brute-force):")
        for r in results:
            preview = r.text[:220].replace("\n", " ")
            logger.info(
                f"{r.score:.4f} | {r.doc_id} p.{r.page} | {r.chunk_id} | {preview}..."
            )

    logger.info("RUNTIME – terminé")


if __name__ == "__main__":
    main()
