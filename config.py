from pathlib import Path

# Dossiers
PDF_DIR = Path("rag_docs")  # les PDFs sont ici
DATA_DIR = Path("data")

# Extraction PDF
MAX_PAGES_PER_PDF = None  # ex: 50 pour limiter, ou None pour tout
MIN_CHARS_PER_PAGE = 20  # ignore pages quasi vides (scan, etc.)

# Chunking (PoC)
# Chunking (PoC)
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 200
CHUNK_SIZE_FINE = 40  # +/- autour de CHUNK_SIZE_CHARS
CHUNK_SOFT_MIN = CHUNK_SIZE_CHARS - CHUNK_SIZE_FINE
CHUNK_SOFT_MAX = CHUNK_SIZE_CHARS + CHUNK_SIZE_FINE

# Embeddings
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_BATCH_SIZE = 32

# Artefacts build (sans FAISS pour l'instant)
ARTIFACTS_DIR = DATA_DIR
CHUNKS_JSONL_PATH = ARTIFACTS_DIR / "chunks.jsonl"
EMBEDDINGS_NPY_PATH = ARTIFACTS_DIR / "embeddings.npy"
CHUNK_IDS_PATH = ARTIFACTS_DIR / "chunk_ids.txt"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

# Reranking
RERANKER_MODEL_NAME = "antoinelouis/crossencoder-mMiniLMv2-L12-mmarcoFR"
DENSE_TOP_K = 30  # candidats brute-force avant rerank
RERANK_TOP_N = 6  # contexte final (plus tard pour LLM)
RERANK_SCORE_MIN = None  # ex: 0.0 ou 0.1 si tu veux filtrer, sinon None


# Logs
VERBOSE = True
