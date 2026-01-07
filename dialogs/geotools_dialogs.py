# geotools_dialogs.py
from __future__ import annotations

import re
from typing import Optional, Tuple

from tools.geo_tools import (
    calculate_distance,
    calculate_flight_time,
    get_gps_coordinates,
)


_RE_ARROW = re.compile(r"\s*(->|→)\s*")
_RE_ENTRE = re.compile(r"\bentre\b", re.IGNORECASE)
_RE_ET = re.compile(r"\bet\b", re.IGNORECASE)
_RE_DE = re.compile(r"\bde\b", re.IGNORECASE)
_RE_A = re.compile(r"\bà\b|\ba\b", re.IGNORECASE)


def is_geo_question(question: str) -> bool:
    q = question.lower()
    return any(
        k in q
        for k in (
            "distance",
            "vol d'oiseau",
            "trajet",
            "durée",
            "temps",
            "km",
            "heures",
        )
    )


def extract_two_places_fr(question: str) -> Optional[Tuple[str, str]]:
    """
    Heuristique simple :
      - "entre X et Y"
      - "de X à Y"
      - "X -> Y" / "X → Y"

    On renvoie (A, B) ou None si non détectable.
    """
    q = " ".join((question or "").strip().split())
    if not q:
        return None

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
    # (on prend la dernière occurrence de "de" pour éviter "durée du trajet de ...")
    q_low = q.lower()
    if " de " in f" {q_low} " and (" à " in f" {q_low} " or " a " in f" {q_low} "):
        try:
            tail = q.rsplit(" de ", 1)[1]
            # split sur " à " si présent, sinon " a "
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

    return None


def try_answer(question: str, speed_kmh_for_duration: float = 80.0) -> Optional[str]:
    """
    Tente de répondre via geo_tools (Nominatim + geodesic).
    Retourne une string si succès, sinon None.

    Remarque : durée = estimation très simple distance/speed_kmh_for_duration.
    """
    if not is_geo_question(question):
        return None

    places = extract_two_places_fr(question)
    if not places:
        return None

    a_name, b_name = places

    a = get_gps_coordinates(a_name)
    b = get_gps_coordinates(b_name)
    if not a.get("ok") or not b.get("ok"):
        return None

    dist = calculate_distance(a["lat"], a["lon"], b["lat"], b["lon"])
    if not dist.get("ok"):
        return None

    distance_km = dist["distance_km"]
    q = question.lower()

    # Si l'utilisateur parle de durée/temps, on fournit une estimation simple.
    if "durée" in q or "temps" in q:
        t = calculate_flight_time(distance_km, speed_kmh_for_duration)
        if t.get("ok"):
            return (
                f"Distance (vol d'oiseau) {a_name} → {b_name} : {distance_km} km.\n"
                f"Estimation de durée à {speed_kmh_for_duration:.0f} km/h : {t['hours']} h."
            )

    return f"Distance (vol d'oiseau) {a_name} → {b_name} : {distance_km} km."
