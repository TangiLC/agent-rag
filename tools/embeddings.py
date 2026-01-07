from __future__ import annotations
import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tools.chunking import ChunkRecord

logger = logging.getLogger(__name__)


def _get_device(device: Optional[str] = None) -> str:
    """
    Détermine le device à utiliser (CPU/GPU).

    Args:
        device: Device explicite ("cpu", "cuda", "mps") ou None pour auto-détection

    Returns:
        Device string ("cpu", "cuda", ou "mps")
    """
    if device is not None:
        return device

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def _to_e5_passage_style(text: str) -> str:
    """
    Formate un texte pour l'encodage en tant que passage (document).
    Les modèles E5 utilisent des préfixes pour améliorer les performances.
    """
    return f"passage: {text}"


def _to_e5_query_style(text: str) -> str:
    """
    Formate un texte pour l'encodage en tant que requête (query).
    Les modèles E5 utilisent des préfixes pour améliorer les performances.
    """
    return f"query: {text}"


def embed_chunks(
    chunks: List[ChunkRecord],
    model_name: str,
    batch_size: int = 32,
    device: Optional[str] = None,
    show_progress: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode une liste de chunks en embeddings (passages).

    Args:
        chunks: Liste des chunks à encoder
        model_name: Nom du modèle SentenceTransformer (ex: "intfloat/multilingual-e5-small")
        batch_size: Taille des batches pour l'encodage
        device: Device à utiliser ("cpu", "cuda", "mps") ou None pour auto-détection
        show_progress: Afficher une barre de progression

    Returns:
        Tuple contenant:
        - embeddings: np.ndarray shape (n, dim) float32 normalisé
        - chunk_ids: List[str] alignée avec embeddings
    """
    if not chunks:
        logger.warning("Liste de chunks vide, retour d'un array vide")
        return np.zeros((0, 0), dtype=np.float32), []

    # Détection du device
    selected_device = _get_device(device)
    logger.info(f"Device sélectionné: {selected_device}")

    # Chargement du modèle
    logger.info(f"Chargement du modèle d'embedding: {model_name}")
    model = SentenceTransformer(model_name, device=selected_device)

    # Préparation des données
    texts = [_to_e5_passage_style(c.text) for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    logger.info(
        f"Encodage de {len(texts)} passages "
        f"(batch_size={batch_size}, device={selected_device})"
    )

    # Encodage des embeddings
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Important pour cosine similarity avec FAISS IP
    )

    # Conversion en float32 pour FAISS
    vectors = np.asarray(vectors, dtype=np.float32)

    # Validation
    if vectors.shape[0] != len(chunk_ids):
        raise ValueError(
            f"Mismatch entre embeddings et IDs: "
            f"{vectors.shape[0]} embeddings vs {len(chunk_ids)} IDs"
        )

    logger.info(f"Embeddings générés: shape={vectors.shape}, dtype={vectors.dtype}")
    return vectors, chunk_ids


def embed_query(
    query: str,
    model_name: str,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Encode une requête utilisateur en embedding.
    À utiliser pour des appels ponctuels. Pour des boucles, préférer embed_query_with_model().

    Args:
        query: Texte de la requête utilisateur
        model_name: Nom du modèle SentenceTransformer
        device: Device à utiliser ("cpu", "cuda", "mps") ou None pour auto-détection

    Returns:
        Embedding normalisé shape (dim,) en float32
    """
    if not query or not query.strip():
        raise ValueError("La requête ne peut pas être vide")

    # Détection du device
    selected_device = _get_device(device)

    # Chargement du modèle
    logger.info(
        f"Chargement du modèle pour la requête: {model_name} (device={selected_device})"
    )
    model = SentenceTransformer(model_name, device=selected_device)

    # Formatage et encodage
    formatted_query = _to_e5_query_style(query)

    vector = model.encode(
        formatted_query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Conversion en float32
    vector = np.asarray(vector, dtype=np.float32)

    logger.info(f"Requête encodée: shape={vector.shape}, dtype={vector.dtype}")
    return vector


def embed_query_with_model(
    query: str,
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Encode une requête avec un modèle déjà chargé (optimisé pour boucles/runtime).
    Cette fonction est recommandée pour les agents qui traitent plusieurs requêtes.

    Args:
        query: Texte de la requête utilisateur
        model: Instance SentenceTransformer pré-chargée

    Returns:
        Embedding normalisé shape (dim,) en float32
    """
    if not query or not query.strip():
        raise ValueError("La requête ne peut pas être vide")

    # Formatage avec préfixe E5
    formatted_query = _to_e5_query_style(query)

    # Encodage
    vector = model.encode(
        formatted_query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(vector, dtype=np.float32)


def embed_queries_batch(
    queries: List[str],
    model_name: str,
    batch_size: int = 32,
    device: Optional[str] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode plusieurs requêtes en batch.

    Args:
        queries: Liste des requêtes à encoder
        model_name: Nom du modèle SentenceTransformer
        batch_size: Taille des batches pour l'encodage
        device: Device à utiliser ("cpu", "cuda", "mps") ou None pour auto-détection
        show_progress: Afficher une barre de progression

    Returns:
        Embeddings normalisés shape (n, dim) en float32
    """
    if not queries:
        logger.warning("Liste de requêtes vide, retour d'un array vide")
        return np.zeros((0, 0), dtype=np.float32)

    # Validation
    if any(not q or not q.strip() for q in queries):
        raise ValueError("Toutes les requêtes doivent être non-vides")

    # Détection du device
    selected_device = _get_device(device)
    logger.info(f"Device sélectionné: {selected_device}")

    # Chargement du modèle
    logger.info(f"Chargement du modèle: {model_name}")
    model = SentenceTransformer(model_name, device=selected_device)

    # Formatage des requêtes
    formatted_queries = [_to_e5_query_style(q) for q in queries]

    logger.info(
        f"Encodage de {len(formatted_queries)} requêtes "
        f"(batch_size={batch_size}, device={selected_device})"
    )

    # Encodage
    vectors = model.encode(
        formatted_queries,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Conversion en float32
    vectors = np.asarray(vectors, dtype=np.float32)

    logger.info(f"Requêtes encodées: shape={vectors.shape}, dtype={vectors.dtype}")
    return vectors


def embed_queries_batch_with_model(
    queries: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode plusieurs requêtes en batch avec un modèle pré-chargé.

    Args:
        queries: Liste des requêtes à encoder
        model: Instance SentenceTransformer pré-chargée
        batch_size: Taille des batches pour l'encodage
        show_progress: Afficher une barre de progression

    Returns:
        Embeddings normalisés shape (n, dim) en float32
    """
    if not queries:
        logger.warning("Liste de requêtes vide, retour d'un array vide")
        return np.zeros((0, 0), dtype=np.float32)

    # Validation
    if any(not q or not q.strip() for q in queries):
        raise ValueError("Toutes les requêtes doivent être non-vides")

    # Formatage des requêtes avec préfixe E5
    formatted_queries = [_to_e5_query_style(q) for q in queries]

    # Encodage
    vectors = model.encode(
        formatted_queries,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(vectors, dtype=np.float32)
