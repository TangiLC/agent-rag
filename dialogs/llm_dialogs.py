# llm_dialogs.py
from __future__ import annotations

from typing import List, Optional, Tuple

from rag_agent_init import AgentState


def build_rag_messages(
    question: str,
    contexts: List[str],
    bird_name: Optional[str] = None,
) -> Tuple[str, str]:
    joined = "\n\n---\n\n".join([c.strip() for c in contexts if c and c.strip()])

    bird_instruction = ""
    if bird_name:
        bird_instruction = (
            f"S'il existe dans le CONTEXTE des informations factuelles sur le {bird_name}, "
            f"ajoute UNE phrase finale commençant par "
            f'"Information sur le {bird_name} :" et résumant ces informations.\n'
        )

    system = (
        "Tu es un assistant RAG. Règles strictes :\n"
        "1) Utilise uniquement les informations du CONTEXTE.\n"
        "2) Si l'information n'est pas dans le CONTEXTE, dis exactement : \"Je ne sais pas d'après le corpus.\".\n"
        "3) Réponds de façon courte et directe.\n"
        f"{bird_instruction}"
    )

    user = "CONTEXTE:\n" f"{joined}\n\n" "QUESTION:\n" f"{question.strip()}\n"
    return system, user


def answer_from_contexts(
    state: AgentState,
    question: str,
    contexts: List[str],
    bird_name: Optional[str] = None,
) -> str:
    """
    Appelle llama-server via state.llama_server (OpenAI-compatible /v1/chat/completions).
    """
    if not getattr(state, "llama_server", None):
        raise RuntimeError("llama_server non initialisé (state.llama_server est None).")

    s = state.settings
    system, user = build_rag_messages(
        question=question, contexts=contexts, bird_name=bird_name
    )

    txt = state.llama_server.chat_with_feedback(
        system=system,
        user=user,
        max_tokens=s.llama_max_tokens,
        temperature=s.llama_temperature,
        top_p=s.llama_top_p,
        repeat_penalty=s.llama_repeat_penalty,
        timeout_s=s.llama_request_timeout_s,
        show_waiting=True,
    )
    txt = (txt or "").strip()
    return txt or "Je ne sais pas d'après le corpus."
