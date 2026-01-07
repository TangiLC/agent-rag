import logging

import config
from tools.retrieval_bruteforce import (
    embed_query,
    load_chunk_ids,
    load_chunk_store,
    load_embeddings,
    top_k_by_dot,
)
from tools.reranker import rerank


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

    # Artefacts
    if not (
        config.CHUNKS_JSONL_PATH.exists()
        and config.CHUNK_IDS_PATH.exists()
        and config.EMBEDDINGS_NPY_PATH.exists()
    ):
        logger.info("Artefacts manquants. Lance: python build_index.py")
        return

    logger.info("Chargement artefacts")
    chunk_store = load_chunk_store(config.CHUNKS_JSONL_PATH)
    chunk_ids = load_chunk_ids(config.CHUNK_IDS_PATH)
    embeddings = load_embeddings(config.EMBEDDINGS_NPY_PATH)

    if len(chunk_ids) != embeddings.shape[0]:
        logger.info(
            f"Incohérence: {len(chunk_ids)} chunk_ids vs {embeddings.shape[0]} embeddings"
        )
        logger.info("Lance: python build_index.py")
        return

    logger.info(
        f"OK artefacts: {embeddings.shape[0]} chunks, dim={embeddings.shape[1]}"
    )

    # Modèle embedding query (chargé une seule fois)
    from sentence_transformers import SentenceTransformer

    logger.info(f"Chargement embedder (requêtes): {config.EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cpu")

    logger.info("Prêt. Tape une question (vide pour quitter).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break

        # 1) encode query
        q_vec = embed_query(embedder, q)

        # 2) dense retrieval
        dense = top_k_by_dot(
            query_vec=q_vec,
            embeddings=embeddings,
            chunk_ids=chunk_ids,
            top_k=config.DENSE_TOP_K,
        )

        # 3) build candidates for rerank
        cand_ids = [cid for cid, _ in dense]
        cand_texts = [
            chunk_store[cid]["text"] for cid in cand_ids if cid in chunk_store
        ]

        # réalignement au cas où un cid manque du store
        cand_ids = [cid for cid in cand_ids if cid in chunk_store]

        if not cand_ids:
            logger.info("Aucun candidat.")
            continue

        # 4) rerank
        reranked = rerank(
            query=q,
            chunk_texts=cand_texts,
            chunk_ids=cand_ids,
            model_name=config.RERANKER_MODEL_NAME,
        )

        # 5) keep top_n (+ seuil optionnel)
        top = reranked[: config.RERANK_TOP_N]
        if config.RERANK_SCORE_MIN is not None:
            top = [x for x in top if x.score >= config.RERANK_SCORE_MIN]

        if not top:
            logger.info("Non trouvé dans le corpus (après rerank).")
            continue

        logger.info("Top résultats (rerank):")
        for item in top:
            meta = chunk_store[item.chunk_id]
            preview = meta["text"][:220].replace("\n", " ")
            logger.info(
                f"{item.score:.4f} | {meta['doc_id']} p.{meta['page']} | {item.chunk_id} | {preview}..."
            )

    logger.info("RUNTIME – terminé")


if __name__ == "__main__":
    main()
