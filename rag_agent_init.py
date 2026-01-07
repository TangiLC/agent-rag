# rag_agent_init.py
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import faiss  # type: ignore
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)


# -----------------------------
# Config helpers (robuste)
# -----------------------------
def _cfg(name: str, default):
    import config  # import local

    return getattr(config, name, default)


def _path_cfg(name: str, default: str) -> Path:
    val = _cfg(name, None)
    if val is None:
        return Path(default)
    if isinstance(val, Path):
        return val
    return Path(str(val))


# -----------------------------
# Artefacts loaders
# -----------------------------
def load_chunk_store(chunks_jsonl_path: Path) -> Dict[str, dict]:
    if not chunks_jsonl_path.exists():
        raise FileNotFoundError(f"Chunks JSONL introuvable: {chunks_jsonl_path}")

    store: Dict[str, dict] = {}
    with chunks_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            if cid:
                store[cid] = rec
    return store


def load_chunk_ids(chunk_ids_path: Path) -> List[str]:
    if not chunk_ids_path.exists():
        raise FileNotFoundError(f"chunk_ids introuvable: {chunk_ids_path}")

    ids: List[str] = []
    with chunk_ids_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def load_embeddings(embeddings_npy_path: Path) -> np.ndarray:
    if not embeddings_npy_path.exists():
        raise FileNotFoundError(f"Embeddings NPY introuvable: {embeddings_npy_path}")
    arr = np.load(str(embeddings_npy_path))
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def load_faiss_index(index_path: Path) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"Index FAISS introuvable: {index_path}")
    return faiss.read_index(str(index_path))


def pick_device_for_reranker() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class AgentSettings:
    # retrieval/rerank
    faiss_top_k: int
    rerank_min_top1: float

    # answer shaping
    extract_top_n: int
    extract_max_chars: int

    # embeddings
    embed_model_name: str
    e5_query_prefix: str

    # reranker
    reranker_model_name: str
    reranker_device: str

    # llm llama.cpp
    use_llm: bool
    llama_cli_path: str
    llama_model_path: Path
    llama_n_predict: int
    llama_ctx_size: int
    llama_temperature: float
    llama_top_p: float
    llama_repeat_penalty: float
    llama_threads: Optional[int]
    llama_timeout_s: int


@dataclass
class AgentState:
    settings: AgentSettings
    chunk_store: Dict[str, dict]
    chunk_ids: List[str]
    index: faiss.Index
    embedder: SentenceTransformer
    reranker: CrossEncoder


def init_state() -> AgentState:
    # --- Artefacts
    data_dir = _path_cfg("DATA_DIR", "data")
    artefacts_dir = _path_cfg("ARTIFACTS_DIR", str(data_dir))

    chunks_jsonl_path = _path_cfg(
        "CHUNKS_JSONL_PATH", str(artefacts_dir / "chunks.jsonl")
    )
    chunk_ids_path = _path_cfg("CHUNK_IDS_PATH", str(artefacts_dir / "chunk_ids.txt"))
    embeddings_npy_path = _path_cfg(
        "EMBEDDINGS_NPY_PATH", str(artefacts_dir / "embeddings.npy")
    )
    faiss_index_path = _path_cfg("FAISS_INDEX_PATH", str(artefacts_dir / "faiss.index"))

    # --- Retrieval / rerank / guardrail
    faiss_top_k = int(_cfg("FAISS_TOP_K", _cfg("DENSE_TOP_K", 20)))
    rerank_min_top1 = float(_cfg("RERANK_MIN_TOP1_SCORE", 0.2))

    # --- Answer shaping
    extract_top_n = int(_cfg("EXTRACTIVE_TOP_N", 4))
    extract_max_chars = int(_cfg("EXTRACTIVE_MAX_CHARS_PER_CHUNK", 800))

    # --- Embeddings
    embed_model_name = str(
        _cfg("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
    )
    e5_query_prefix = str(_cfg("E5_QUERY_PREFIX", "query: "))

    # --- Reranker
    reranker_model_name = str(
        _cfg("RERANKER_MODEL_NAME", "antoinelouis/crossencoder-mMiniLMv2-L12-mmarcoFR")
    )
    reranker_device = pick_device_for_reranker()

    # --- LLM llama.cpp
    use_llm = bool(_cfg("USE_LLM", True))
    llama_cli_path = str(_cfg("LLAMA_CLI_PATH", "llama-cli"))
    llama_model_path = _path_cfg(
        "LLAMA_MODEL_PATH", "models/llama-3.2-3b-instruct-q4_k_m.gguf"
    )
    llama_n_predict = int(_cfg("LLAMA_N_PREDICT", 200))
    llama_ctx_size = int(_cfg("LLAMA_CTX_SIZE", 4096))
    llama_temperature = float(_cfg("LLAMA_TEMPERATURE", 0.2))
    llama_top_p = float(_cfg("LLAMA_TOP_P", 0.9))
    llama_repeat_penalty = float(_cfg("LLAMA_REPEAT_PENALTY", 1.1))
    llama_timeout_s = int(_cfg("LLAMA_TIMEOUT_S", 120))

    threads_val = _cfg("LLAMA_THREADS", None)
    llama_threads: Optional[int] = (
        int(threads_val) if isinstance(threads_val, int) else None
    )

    logger.info("INIT – chargement artefacts")
    chunk_store = load_chunk_store(chunks_jsonl_path)
    chunk_ids = load_chunk_ids(chunk_ids_path)
    _ = load_embeddings(embeddings_npy_path)  # garde-fou: existence / format
    index = load_faiss_index(faiss_index_path)

    logger.info("INIT – chargement embedder")
    embedder = SentenceTransformer(embed_model_name)

    logger.info("INIT – chargement reranker")
    reranker = CrossEncoder(reranker_model_name, device=reranker_device)

    settings = AgentSettings(
        faiss_top_k=faiss_top_k,
        rerank_min_top1=rerank_min_top1,
        extract_top_n=extract_top_n,
        extract_max_chars=extract_max_chars,
        embed_model_name=embed_model_name,
        e5_query_prefix=e5_query_prefix,
        reranker_model_name=reranker_model_name,
        reranker_device=reranker_device,
        use_llm=use_llm,
        llama_cli_path=llama_cli_path,
        llama_model_path=llama_model_path,
        llama_n_predict=llama_n_predict,
        llama_ctx_size=llama_ctx_size,
        llama_temperature=llama_temperature,
        llama_top_p=llama_top_p,
        llama_repeat_penalty=llama_repeat_penalty,
        llama_threads=llama_threads,
        llama_timeout_s=llama_timeout_s,
    )

    # Vérification explicite du modèle si LLM activé
    if settings.use_llm and not settings.llama_model_path.exists():
        raise FileNotFoundError(f"Modèle GGUF introuvable: {settings.llama_model_path}")

    return AgentState(
        settings=settings,
        chunk_store=chunk_store,
        chunk_ids=chunk_ids,
        index=index,
        embedder=embedder,
        reranker=reranker,
    )
