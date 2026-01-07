# rag_agent_dialog.py
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from rag_agent_init import AgentState

logger = logging.getLogger(__name__)

# ============================
# Routing (minimal, sans sur-automation)
# ============================

_GEO_HINTS = re.compile(
    r"\b(ville|distance|km|kilom[eè]tres?|trajet|vol|temps de vol|entre)\b",
    re.IGNORECASE,
)

_TWO_CITIES = re.compile(
    r"\bentre\s+(?P<a>.+?)\s+et\s+(?P<b>.+?)(?:\?|\.|$)",
    re.IGNORECASE,
)

_SPEED_KMH = re.compile(r"(\d{2,4})\s*km\s*/\s*h", re.IGNORECASE)


@dataclass(frozen=True)
class RouteDecision:
    use_geo: bool
    use_rag: bool


def decide_route(question: str) -> RouteDecision:
    q = (question or "").strip()
    if not q:
        return RouteDecision(use_geo=False, use_rag=False)
    use_geo = bool(_GEO_HINTS.search(q))
    return RouteDecision(use_geo=use_geo, use_rag=True)


def _extract_two_cities(question: str) -> Optional[Tuple[str, str]]:
    m = _TWO_CITIES.search(question or "")
    if not m:
        return None
    a = (m.group("a") or "").strip()
    b = (m.group("b") or "").strip()
    if not a or not b:
        return None
    return a, b


def _extract_speed_kmh(question: str) -> Optional[float]:
    m = _SPEED_KMH.search(question or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# ============================
# GEO tools (optionnels)
# ============================


def _geo_distance_part(question: str) -> Optional[str]:
    """
    Calcule une distance uniquement si le pattern est explicite: "entre A et B".
    N'appelle pas Nominatim si l'extraction n'est pas claire.
    """
    cities = _extract_two_cities(question)
    if not cities:
        return None

    # Imports ici pour garder le module utilisable même sans geopy installé
    try:
        from tools.geo_tools import (
            get_gps_coordinates,
            calculate_distance,
            calculate_flight_time,
        )
    except Exception as e:
        logger.info(f"GEO – tools indisponibles: {e}")
        return None

    a, b = cities
    ga = get_gps_coordinates(a)
    gb = get_gps_coordinates(b)

    if not ga.get("ok") or not gb.get("ok"):
        return None

    dist = calculate_distance(ga["lat"], ga["lon"], gb["lat"], gb["lon"])
    if not dist.get("ok"):
        return None

    distance_km = dist["distance_km"]
    out = f"Distance (vol d'oiseau) {a} → {b} : {distance_km} km."

    speed = _extract_speed_kmh(question)
    if speed is not None and speed > 0:
        t = calculate_flight_time(distance_km, speed)
        if t.get("ok"):
            out += f" Temps à {int(speed)} km/h : {t['hours']} h."

    return out


# ============================
# RAG core (FAISS -> rerank -> LLM -> sources)
# ============================


@dataclass(frozen=True)
class RerankedItem:
    chunk_id: str
    score: float


def embed_query(state: AgentState, query: str) -> np.ndarray:
    text = f"{state.settings.e5_query_prefix}{query}"
    vec = state.embedder.encode(
        [text], normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vec, dtype=np.float32)


def rerank(
    state: AgentState, query: str, candidates: Sequence[Tuple[str, str]]
) -> List[RerankedItem]:
    if not candidates:
        return []
    pairs = [(query, txt) for _, txt in candidates]
    scores = state.reranker.predict(pairs)
    items = [
        RerankedItem(chunk_id=candidates[i][0], score=float(scores[i]))
        for i in range(len(candidates))
    ]
    items.sort(key=lambda x: x.score, reverse=True)
    return items


def build_rag_prompt(question: str, contexts: List[str]) -> str:
    joined = "\n\n---\n\n".join(contexts)
    return (
        "SYSTEM:\n"
        "Tu es un assistant RAG. Règles strictes :\n"
        "1) Utilise uniquement les informations du CONTEXTE.\n"
        "2) Si l'information n'est pas dans le CONTEXTE, dis exactement : \"Je ne sais pas d'après le corpus.\".\n"
        "3) Réponds de façon courte et directe.\n\n"
        "CONTEXTE:\n"
        f"{joined}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "RÉPONSE:\n"
    )


def run_llama_cli(state: AgentState, prompt: str) -> str:
    s = state.settings
    cmd: List[str] = [
        s.llama_cli_path,
        "-m",
        str(s.llama_model_path),
        "-p",
        prompt,
        "-n",
        str(s.llama_n_predict),
        "-c",
        str(s.llama_ctx_size),
        "--temp",
        str(s.llama_temperature),
        "--top-p",
        str(s.llama_top_p),
        "--repeat-penalty",
        str(s.llama_repeat_penalty),
        "--no-display-prompt",
    ]
    if s.llama_threads is not None and s.llama_threads > 0:
        cmd += ["-t", str(s.llama_threads)]

    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=s.llama_timeout_s
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"llama-cli a échoué (code={proc.returncode}). stderr: {stderr[:800]}"
        )
    return (proc.stdout or "").strip()


def format_sources(state: AgentState, items: Sequence[RerankedItem]) -> str:
    lines = ["Sources:"]
    for it in items:
        meta = state.chunk_store.get(it.chunk_id, {})
        doc_id = meta.get("doc_id", "?")
        page = meta.get("page", "?")
        chunk_id = meta.get("chunk_id", it.chunk_id)
        lines.append(f"- {doc_id} / p.{page} / {chunk_id}")
    return "\n".join(lines)


def rag_answer_once(state: AgentState, question: str) -> str:
    s = state.settings

    q_vec = embed_query(state, question)
    _, idxs = state.index.search(q_vec, s.faiss_top_k)

    cand_ids: List[str] = []
    for j in idxs[0].tolist():
        if j is None or int(j) < 0:
            continue
        j = int(j)
        if j >= len(state.chunk_ids):
            continue
        cand_ids.append(state.chunk_ids[j])

    candidates: List[Tuple[str, str]] = []
    for cid in cand_ids:
        rec = state.chunk_store.get(cid)
        if not rec:
            continue
        txt = rec.get("text", "")
        if txt:
            candidates.append((cid, txt))

    if not candidates:
        return "Non trouvé dans le corpus."

    reranked = rerank(state, question, candidates)
    if not reranked:
        return "Non trouvé dans le corpus."

    if reranked[0].score < s.rerank_min_top1:
        return "Non trouvé dans le corpus."

    k = min(s.extract_top_n, len(reranked))
    selected = reranked[:k]

    contexts: List[str] = []
    for it in selected:
        meta = state.chunk_store.get(it.chunk_id, {})
        txt = str(meta.get("text", ""))[: s.extract_max_chars]
        contexts.append(txt)

    if s.use_llm:
        prompt = build_rag_prompt(question=question, contexts=contexts)
        try:
            answer = run_llama_cli(state, prompt).strip()
        except Exception as e:
            logger.info(f"LLM – échec, fallback extractif: {e}")
            answer = contexts[0].strip() if contexts else "Non trouvé dans le corpus."
    else:
        answer = contexts[0].strip() if contexts else "Non trouvé dans le corpus."

    return answer + "\n\n" + format_sources(state, selected)


# ============================
# Public API: combine tools + RAG
# ============================


def answer_once(state: AgentState, question: str) -> str:
    decision = decide_route(question)
    if not (decision.use_geo or decision.use_rag):
        return ""

    parts: List[str] = []

    geo_part = None
    if decision.use_geo:
        geo_part = _geo_distance_part(question)
        if geo_part:
            parts.append(geo_part)

    if decision.use_rag:
        rag_part = rag_answer_once(state, question)

        # Si GEO a répondu, éviter le "faux échec" visuel
        if geo_part and rag_part.strip() == "Non trouvé dans le corpus.":
            pass
        else:
            parts.append(rag_part)

    return "\n\n".join([p for p in parts if p.strip()])


def repl(state: AgentState) -> None:
    print("\nRAG prêt. Entrez votre question (ou 'exit').\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            break

        out = answer_combined_geo_plus_rag(state, q)
        print()
        print(out)
        print()


def _extract_between_cities(question: str) -> Optional[Tuple[str, str]]:
    m = _TWO_CITIES.search(question or "")
    if not m:
        return None
    a = (m.group("a") or "").strip()
    b = (m.group("b") or "").strip()
    if not a or not b:
        return None
    return a, b


def _build_rag_subqueries(
    question: str, a: Optional[str], b: Optional[str], max_q: int = 3
) -> List[str]:
    # Règles simples : question puis A puis B, sans doublons
    qs: List[str] = []

    def add(x: str) -> None:
        x = (x or "").strip()
        if not x:
            return
        if x.lower() in {q.lower() for q in qs}:
            return
        qs.append(x)

    add(question)
    if a:
        add(a)
    if b:
        add(b)

    return qs[:max_q]


def _strip_sources_block(text: str) -> str:
    # Coupe à partir de "Sources:" si présent
    idx = text.find("\n\nSources:")
    if idx >= 0:
        return text[:idx].strip()
    idx2 = text.find("\nSources:")
    if idx2 >= 0:
        return text[:idx2].strip()
    return text.strip()


def _extract_sources_lines(text: str) -> List[str]:
    # Récupère les lignes "- doc / p.X / chunk"
    lines = []
    if "Sources:" not in text:
        return lines
    part = text.split("Sources:", 1)[1]
    for raw in part.splitlines():
        s = raw.strip()
        if s.startswith("- "):
            lines.append(s)
    return lines


def answer_combined_geo_plus_rag(state: AgentState, question: str) -> str:
    parts: List[str] = []

    # 1) GEO (si possible)
    a = b = None
    cities = _extract_between_cities(question)
    geo_part = None
    if cities:
        a, b = cities
        geo_part = _geo_distance_part(question)
        if geo_part:
            parts.append(geo_part)

    # 2) RAG multi-queries (question + A + B)
    subqueries = _build_rag_subqueries(question, a, b, max_q=3)

    corpus_sections: List[str] = []
    all_sources: List[str] = []
    seen_sources = set()

    for sq in subqueries:
        rag_out = rag_answer_once(state, sq)

        # Si la sous-question ne trouve rien, on ignore (sauf si c'est la question principale)
        if rag_out.strip() == "Non trouvé dans le corpus." and sq != question:
            continue

        answer_txt = _strip_sources_block(rag_out)
        src_lines = _extract_sources_lines(rag_out)

        # Construire une mini-section
        if sq == question:
            title = "Corpus (question)"
        else:
            title = f"Corpus ({sq})"

        if answer_txt.strip() != "Non trouvé dans le corpus.":
            corpus_sections.append(f"{title}:\n{answer_txt}")

        for s in src_lines:
            if s not in seen_sources:
                seen_sources.add(s)
                all_sources.append(s)

    if corpus_sections:
        parts.append("\n\n".join(corpus_sections))
    elif not geo_part:
        # Ni GEO ni corpus: message standard
        return "Non trouvé dans le corpus."

    # 3) Sources globales à la fin (format strict)
    if all_sources:
        parts.append("Sources:\n" + "\n".join(all_sources))

    return "\n\n".join([p for p in parts if p.strip()])
