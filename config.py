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
CHUNK_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 200
CHUNK_SIZE_FINE = 60  # +/- autour de CHUNK_SIZE_CHARS
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

# Guardrails "hors corpus" (à calibrer)
# B1: si top1 < seuil => "non trouvé"
RERANK_MIN_TOP1_SCORE = 0.2
RERANK_SCORE_MIN = None

# Réponse extractive (sans LLM)
EXTRACTIVE_TOP_N = 4
EXTRACTIVE_MAX_CHARS_PER_CHUNK = 800

# FAISS
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
FAISS_USE_IP = True  # embeddings normalisés -> cosine == inner product

# Dense retrieval via FAISS
FAISS_TOP_K = DENSE_TOP_K  # même valeur que ta config actuelle

# LLM (llama.cpp)
USE_LLM = True
LLAMA_CLI_PATH = "./llama.cpp/llama-cli"
LLAMA_MODEL_PATH = Path("models/llama-3.2-3b-instruct-q4_k_m.gguf")

# Génération
LLAMA_N_PREDICT = 200
LLAMA_CTX_SIZE = 4096
LLAMA_TEMPERATURE = 0.2
LLAMA_TOP_P = 0.9
LLAMA_REPEAT_PENALTY = 1.1
LLAMA_THREADS = None  # ex: 8, ou None pour ne pas forcer
LLAMA_TIMEOUT_S = 120


# Logs
VERBOSE = True
