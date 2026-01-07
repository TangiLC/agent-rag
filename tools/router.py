# tools/router.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_CITY_HINTS = re.compile(
    r"\b(ville|distance|km|kilom[eè]tres?|trajet|vol|temps de vol|entre)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    use_geo: bool
    use_rag: bool


def decide_route(question: str) -> RouteDecision:
    """
    Routeur minimal:
    - GEO si on détecte des indices évidents (distance/ville/km/entre...)
    - RAG par défaut (car ton corpus est la source principale)
    """
    q = (question or "").strip()
    if not q:
        return RouteDecision(use_geo=False, use_rag=False)

    geo = bool(_CITY_HINTS.search(q))
    rag = True  # par défaut: toujours tenter RAG, puis combiner si GEO
    return RouteDecision(use_geo=geo, use_rag=rag)
