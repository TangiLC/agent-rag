from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from rag_agent_init import AgentState
from dialogs.llm_dialogs import answer_from_contexts
from tools.geo_tools import (
    get_gps_coordinates,
    calculate_distance,
    calculate_flight_time,
)
from tools.prompt_interpreter import interpret_query

logger = logging.getLogger(__name__)

# ============================
# Heuristiques (conservées pour fallback potentiel)
# ============================

_BIRD_NAMES = re.compile(
    r"\b(aigle|albatros|faucon|gerfaut|pelerin|pélerin|fregate|frégate|"
    r"fuligule|goeland|goéland|harle|martinet|oie|oiseau)\b",
    re.IGNORECASE,
)

_SPEED_KMH = re.compile(r"(\d{1,3})\s*km\s*/\s*h", re.IGNORECASE)


# ============================
# Retrieval / Rerank (repris du RAG)
# ============================


@dataclass(frozen=True)
class RetrievedBlock:
    chunk_id: str
    text: str
    score: float


def embed_query(state: AgentState, query: str) -> np.ndarray:
    text = f"{state.settings.e5_query_prefix}{query}"
    vec = state.embedder.encode(
        [text], normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vec, dtype=np.float32)


def faiss_retrieve(state: AgentState, qvec: np.ndarray, top_k: int) -> List[str]:
    _, idxs = state.index.search(qvec, top_k)
    out: List[str] = []
    for j in idxs[0].tolist():
        if j is None or j < 0 or j >= len(state.chunk_ids):
            continue
        out.append(state.chunk_ids[int(j)])
    return out


def rerank(
    state: AgentState, query: str, chunk_ids: Sequence[str]
) -> List[RetrievedBlock]:
    pairs = []
    valid_chunk_ids = []

    for cid in chunk_ids:
        rec = state.chunk_store.get(cid)
        if rec and rec.get("text"):
            pairs.append((query, rec["text"]))
            valid_chunk_ids.append(cid)

    if not pairs:
        return []

    scores = state.reranker.predict(pairs)

    items: List[RetrievedBlock] = []
    for i, cid in enumerate(valid_chunk_ids):
        rec = state.chunk_store.get(cid)
        if rec and rec.get("text"):
            items.append(
                RetrievedBlock(
                    chunk_id=cid,
                    text=rec["text"],
                    score=float(scores[i]),
                )
            )

    items.sort(key=lambda x: x.score, reverse=True)
    return items


# ============================
# GEO
# ============================


def _geo_distance_from_names(
    a_name: str, b_name: str
) -> Tuple[Optional[float], Optional[str], Tuple[str, str]]:
    """
    Calcule la distance entre deux villes.

    Returns:
        (distance_km, texte_formaté, (ville_a, ville_b))
    """
    ga = get_gps_coordinates(a_name)
    gb = get_gps_coordinates(b_name)

    if not ga.get("ok") or not gb.get("ok"):
        return None, None, (a_name, b_name)

    dist = calculate_distance(ga["lat"], ga["lon"], gb["lat"], gb["lon"])
    if not dist.get("ok"):
        return None, None, (a_name, b_name)

    km = float(dist["distance_km"])
    txt = f"📍 Distance (vol d'oiseau) {a_name} → {b_name} : {km} km."
    return km, txt, (a_name, b_name)


# ============================
# Oiseau : vitesse + info
# ============================


def _retrieve_blocks(state: AgentState, query: str) -> List[RetrievedBlock]:
    qvec = embed_query(state, query)
    cids = faiss_retrieve(state, qvec, state.settings.faiss_top_k)
    ranked = rerank(state, query, cids)
    if not ranked or ranked[0].score < state.settings.rerank_min_top1:
        return []
    return ranked[: state.settings.extract_top_n]


def _extract_speed(blocks: Sequence[RetrievedBlock], bird_name: str) -> Optional[float]:
    """
    Extrait la vitesse d'un oiseau depuis les blocs RAG.

    Stratégie multi-niveaux :
    1. Recherche ligne contenant le nom + vitesse (format tableau PDF)
    2. Recherche proximité nom → vitesse (240 caractères)
    3. Recherche élargie avec premier mot du nom (ex: "faucon" si "faucon pelerin")
    4. Fallback : première vitesse valide trouvée
    """
    joined = " ".join(b.text for b in blocks if b.text)
    # Normalisation : ajoute espace avant km/h si manquant
    joined = re.sub(r"(\d+)(km/h)", r"\1 km/h", joined, flags=re.IGNORECASE)

    if not bird_name:
        # Pas de nom → retourne première vitesse valide trouvée
        m = _SPEED_KMH.search(joined)
        if m:
            v = float(m.group(1))
            if 10 <= v <= 400:
                return v
        return None

    bird_normalized = bird_name.lower().strip()

    # === STRATÉGIE 1 : Ligne de tableau ===
    # Format PDF : "faucon pelerin Falco peregrinus Falconidae 130km/h"
    # Si le nom est sur la même ligne que la vitesse → match très fiable
    lines = joined.split("\n")
    for line in lines:
        line_lower = line.lower()
        if bird_normalized in line_lower:
            speed_match = _SPEED_KMH.search(line)
            if speed_match:
                v = float(speed_match.group(1))
                if 10 <= v <= 400:
                    logger.debug(
                        f"Vitesse trouvée (ligne tableau) : {v} km/h pour {bird_name}"
                    )
                    return v

    # === STRATÉGIE 2 : Proximité dans le texte ===
    # Cherche "nom_oiseau" suivi dans les 240 chars de "XXX km/h"
    pat = re.compile(
        rf"{re.escape(bird_normalized)}.{{0,240}}?(\d{{1,3}})\s*km\s*/\s*h",
        re.IGNORECASE | re.DOTALL,
    )
    m = pat.search(joined)
    if m:
        v = float(m.group(1))
        if 10 <= v <= 400:
            logger.debug(f"Vitesse trouvée (proximité) : {v} km/h pour {bird_name}")
            return v

    # === STRATÉGIE 3 : Recherche avec premier mot uniquement ===
    # Si "faucon pelerin" non trouvé, essaye juste "faucon"
    # Utile si l'utilisateur dit "faucon" mais le PDF a "faucon pelerin"
    if " " in bird_normalized:
        first_word = bird_normalized.split()[0]
        for line in lines:
            line_lower = line.lower()
            # Vérifie que le premier mot est bien au début d'un mot (pas dans "défaucon")
            if re.search(rf"\b{re.escape(first_word)}", line_lower):
                speed_match = _SPEED_KMH.search(line)
                if speed_match:
                    v = float(speed_match.group(1))
                    if 10 <= v <= 400:
                        logger.debug(
                            f"Vitesse trouvée (premier mot '{first_word}') : {v} km/h "
                            f"pour requête '{bird_name}'"
                        )
                        return v

    # === STRATÉGIE 4 : Fallback générique ===
    # Aucun match spécifique → prend la première vitesse valide du contexte
    m_fallback = _SPEED_KMH.search(joined)
    if m_fallback:
        v = float(m_fallback.group(1))
        if 10 <= v <= 120:
            logger.warning(
                f"Aucun match précis pour '{bird_name}', "
                f"utilisation vitesse générique : {v} km/h"
            )
            return v

    return 42


def _format_sources(state: AgentState, chunk_ids: Sequence[str]) -> str:
    lines = []
    seen = set()
    for cid in chunk_ids:
        if cid in seen:
            continue
        seen.add(cid)
        meta = state.chunk_store.get(cid, {})
        lines.append(
            f"- {meta.get('doc_id','?')} / p.{meta.get('page','?')} / {meta.get('chunk_id',cid)}"
        )
    return "Sources:\n" + "\n".join(lines) if lines else ""


# ============================
# API publique (MODIFIÉE)
# ============================


def answer_once(state: AgentState, question: str) -> str:
    """
    Point d'entrée principal utilisant l'interprétation LLM.
    """
    q = (question or "").strip()
    if not q:
        return ""

    # ============================================================
    # ÉTAPE 1 : Interprétation de la requête par le LLM
    # ============================================================
    logger.info(f"Interprétation de la requête : {q}")
    interpreted = interpret_query(state, q)

    if interpreted is None:
        logger.error("Échec de l'interprétation LLM")
        return "⚠️ Erreur lors de l'analyse de votre question. Veuillez réessayer."

    if not interpreted.is_valid():
        logger.info("Requête invalide (ni villes ni oiseau)")
        return "❌ Pas d'info à ce sujet dans le corpus."

    logger.info(
        f"Interprété → Villes: ({interpreted.ville_a}, {interpreted.ville_b}), "
        f"Oiseau demandé: {interpreted.bird.request}, "
        f"Nom: {interpreted.bird.name}"
    )

    # ============================================================
    # ÉTAPE 2 : Préparation des données
    # ============================================================
    parts: List[str] = []
    source_ids: List[str] = []

    distance_km: Optional[float] = None
    geo_txt: Optional[str] = None

    # Détermination de l'oiseau à utiliser
    bird_name = interpreted.get_bird_name(fallback="goéland")
    is_fallback = not interpreted.bird.has_specific_bird()

    logger.info(f"Oiseau utilisé : {bird_name} (fallback={is_fallback})")

    # ============================================================
    # ÉTAPE 3 : Calcul géographique (si villes présentes)
    # ============================================================
    if interpreted.needs_geo_computation():
        distance_km, geo_txt, _ = _geo_distance_from_names(
            interpreted.ville_a, interpreted.ville_b
        )

        if geo_txt:
            parts.append(geo_txt)
            logger.info(f"Distance calculée : {distance_km} km")
        else:
            logger.warning(
                f"Impossible de calculer la distance entre "
                f"{interpreted.ville_a} et {interpreted.ville_b}"
            )
            parts.append(
                f"⚠️ Impossible de localiser les villes "
                f"{interpreted.ville_a} et {interpreted.ville_b}."
            )

    # ============================================================
    # ÉTAPE 4 : Récupération des informations sur l'oiseau
    # ============================================================
    bird_speed: Optional[float] = None
    bird_info: Optional[str] = None

    # Recherche vitesse
    speed_blocks = _retrieve_blocks(state, f"vitesse {bird_name}")
    source_ids += [b.chunk_id for b in speed_blocks]

    if speed_blocks:
        bird_speed = _extract_speed(speed_blocks, bird_name)
        logger.info(f"Vitesse extraite pour {bird_name} : {bird_speed} km/h")
    else:
        logger.warning(f"Aucun bloc RAG trouvé pour vitesse {bird_name}")

    # Si vitesse non trouvée et oiseau = goéland (fallback), utiliser 40 km/h
    if bird_speed is None and bird_name == "goéland" and is_fallback:
        bird_speed = 40.0
        logger.info("Utilisation vitesse fallback goéland : 40 km/h")

    # Recherche info intéressante sur l'oiseau
    info_blocks = _retrieve_blocks(state, f"{bird_name} caractéristiques")
    source_ids += [b.chunk_id for b in info_blocks]

    if state.settings.use_llm:
        try:
            # Utiliser les blocs d'info, sinon les blocs de vitesse
            ctx = [b.text for b in (info_blocks or speed_blocks)]
            if ctx:
                bird_info = answer_from_contexts(
                    state,
                    question=f"Donne une information factuelle intéressante sur le {bird_name}.",
                    contexts=ctx,
                    bird_name=bird_name,
                ).strip()
                logger.info(f"Info oiseau générée : {bird_info[:80]}...")
        except Exception as e:
            logger.warning(f"LLM info oiseau indisponible : {e}")

    # Ajout de l'info oiseau (toujours présente selon specs)
    if bird_info and bird_info != "Je ne sais pas d'après le corpus.":
        parts.append(bird_info)
    else:
        # Fallback si le LLM ne trouve rien
        parts.append(f"ℹ️ Oiseau de référence : {bird_name}")

    # ============================================================
    # ÉTAPE 5 : Calcul du temps de vol (si distance ET vitesse)
    # ============================================================
    if distance_km is not None and bird_speed is not None:
        t = calculate_flight_time(distance_km, bird_speed)
        if t.get("ok"):
            note = " (référence: goéland)" if is_fallback else ""
            parts.append(
                f"⏱️ Temps de vol estimé (à {int(bird_speed)} km/h{note}) : {t['hours']} h"
            )
            logger.info(f"Temps de vol calculé : {t['hours']} h")
        else:
            logger.error(f"Erreur calcul temps de vol : {t.get('error')}")
    elif interpreted.needs_geo_computation() and bird_speed is None:
        # Distance calculée mais pas de vitesse trouvée
        parts.append(f"⚠️ Vitesse de vol du {bird_name} non trouvée dans le corpus.")

    # ============================================================
    # ÉTAPE 6 : Formatage de la réponse finale
    # ============================================================
    if not parts:
        return "❌ Aucune information trouvée."

    # Ajout des sources
    src = _format_sources(state, source_ids)
    if src:
        parts.append(src)

    return "\n\n".join(parts)


# ============================
# REPL
# ============================


def repl(state: AgentState) -> None:
    print(
        "\n🚀 RAG prêt (orchestrateur avec interprétation LLM). Tape 'exit' pour quitter.\n"
    )
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q or q.lower() in {"exit", "quit", "q"}:
            break

        try:
            out = answer_once(state, q)
        except Exception as e:
            logger.exception("Erreur answer_once")
            out = f"Erreur: {e}"

        print()
        print(out)
        print()
