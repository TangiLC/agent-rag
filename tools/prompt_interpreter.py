# tools/prompt_interpreter.py
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from rag_agent_init import AgentState
from tools.prompt_schemas import InterpretedQuery, BirdRequest, parse_interpreted_query

logger = logging.getLogger(__name__)


# ============================
# Fallback heuristique si LLM timeout
# ============================

_RE_ARROW = re.compile(r"\s*(->|→)\s*")
_RE_ENTRE = re.compile(r"\bentre\b", re.IGNORECASE)
_RE_ET = re.compile(r"\bet\b", re.IGNORECASE)
_RE_DE = re.compile(r"\bde\b", re.IGNORECASE)
_RE_A = re.compile(r"\bà\b|\ba\b", re.IGNORECASE)

_BIRD_NAMES_REGEX = re.compile(
    r"\b(aigle|albatros|faucon|gerfaut|pelerin|pélerin|fregate|frégate|"
    r"fuligule|goeland|goéland|harle|martinet|oie|oiseau)\b",
    re.IGNORECASE,
)


def _extract_cities_heuristic(query: str) -> tuple[Optional[str], Optional[str]]:
    """Extraction heuristique des villes (fallback si LLM timeout)"""
    q = " ".join(query.strip().split())
    if not q:
        return None, None

    # X -> Y / X → Y
    m = _RE_ARROW.split(q, maxsplit=1)
    if len(m) == 3:
        a = m[0].strip(" ,;:-")
        b = m[2].strip(" ,;:-")
        if a and b:
            return a, b

    # entre X et Y
    if _RE_ENTRE.search(q) and _RE_ET.search(q):
        try:
            after = re.split(_RE_ENTRE, q, maxsplit=1)[1].strip()
            parts = re.split(_RE_ET, after, maxsplit=1)
            if len(parts) == 2:
                a = parts[0].strip(" ,;:-")
                b = parts[1].strip(" ,;:-")
                if a and b:
                    return a, b
        except Exception:
            pass

    # de X à Y
    q_low = q.lower()
    if " de " in f" {q_low} " and (" à " in f" {q_low} " or " a " in f" {q_low} "):
        try:
            tail = q.rsplit(" de ", 1)[1]
            if " à " in tail:
                a, b = tail.split(" à ", 1)
            else:
                a, b = tail.split(" a ", 1)
            a = a.strip(" ,;:-")
            b = b.strip(" ,;:-")
            if a and b:
                return a, b
        except Exception:
            pass

    return None, None


def _fallback_heuristic_parse(query: str) -> InterpretedQuery:
    """Parse heuristique si LLM indisponible/timeout"""
    ville_a, ville_b = _extract_cities_heuristic(query)

    # Détection oiseau
    m = _BIRD_NAMES_REGEX.search(query)
    bird_name = m.group(0).lower() if m else None
    bird_request = bird_name is not None

    bird = BirdRequest(request=bird_request, name=bird_name)

    logger.warning(
        f"Fallback heuristique utilisé : villes=({ville_a}, {ville_b}), "
        f"oiseau={bird_name}"
    )

    return InterpretedQuery(ville_a=ville_a, ville_b=ville_b, bird=bird)


SYSTEM_PROMPT = """Tu es un parseur JSON. Extrait les villes et l'oiseau de la requête.

RÈGLES :
- Réponds UNIQUEMENT en JSON, sans texte ni markdown
- Format : {"ville_a": "X" ou null, "ville_b": "Y" ou null, "bird": {"request": true/false, "name": "oiseau" ou null}}
- bird.request = true si un oiseau est mentionné, sinon false
- Noms d'oiseaux en minuscules

EXEMPLES :
"Distance Paris Lyon" → {"ville_a": "Paris", "ville_b": "Lyon", "bird": {"request": false, "name": null}}
"Hirondelle de Nice à Lille" → {"ville_a": "Nice", "ville_b": "Lille", "bird": {"request": true, "name": "hirondelle"}}
"Info sur le faucon" → {"ville_a": null, "ville_b": null, "bird": {"request": true, "name": "faucon"}}

Parse :"""


def interpret_query(state: AgentState, user_query: str) -> Optional[InterpretedQuery]:
    """
    Utilise le LLM pour parser la requête utilisateur en structure exploitable.

    Args:
        state: État de l'agent (contient llama_server)
        user_query: Question de l'utilisateur

    Returns:
        InterpretedQuery si succès
        None si échec (LLM indisponible ou erreur de parsing)
    """
    if not state.llama_server:
        logger.error("LLM non disponible, interprétation impossible")
        return None

    try:
        # Appel LLM avec température basse pour déterminisme
        response = state.llama_server.chat_with_feedback(
            system=SYSTEM_PROMPT,
            user=user_query.strip(),
            max_tokens=150,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.1,
            timeout_s=60,
            show_waiting=True,
        )

        logger.debug(f"Réponse LLM brute : {response[:200]}")

        # Nettoyage de la réponse (au cas où le LLM ajoute du texte/markdown)
        response = response.strip()

        # Suppression des balises markdown si présentes
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        # Extraction du JSON si texte avant/après
        # (certains LLM ajoutent parfois du texte malgré les consignes)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            response = json_match.group(0)

        # Parse JSON
        data = json.loads(response)

        # Construction de l'objet InterpretedQuery
        interpreted = parse_interpreted_query(data)

        logger.info(
            f"Interprétation réussie : "
            f"villes=({interpreted.ville_a}, {interpreted.ville_b}), "
            f"oiseau={'demandé:'+interpreted.bird.name if interpreted.bird.request else 'non demandé'}"
        )

        return interpreted

    except json.JSONDecodeError as e:
        logger.error(f"JSON invalide du LLM (position {e.pos}): {response[:300]}")
        return None

    except (KeyError, TypeError) as e:
        logger.error(f"Structure JSON incorrecte : {e}")
        return None

    except Exception as e:
        # Gestion spéciale du timeout : on essaye le fallback heuristique
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            logger.warning(f"Timeout LLM détecté, utilisation du fallback heuristique")
            try:
                return _fallback_heuristic_parse(user_query)
            except Exception as fallback_err:
                logger.error(f"Fallback heuristique échoué : {fallback_err}")
                return None

        logger.exception("Erreur inattendue lors de l'interprétation")
        return None


def interpret_query_safe(state: AgentState, user_query: str) -> InterpretedQuery:
    """
    Version safe qui retourne toujours un InterpretedQuery.
    En cas d'erreur, retourne une structure vide (invalide).

    Utile pour éviter les vérifications None dans le code appelant.
    """
    result = interpret_query(state, user_query)
    if result is None:
        # Retourne une requête vide (sera considérée comme invalide)
        from tools.prompt_schemas import BirdRequest

        return InterpretedQuery(
            ville_a=None, ville_b=None, bird=BirdRequest(request=False, name=None)
        )
    return result
