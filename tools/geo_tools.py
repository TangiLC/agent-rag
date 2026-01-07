from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from geopy.distance import geodesic
from geopy.geocoders import Nominatim


# Cache mémoire simple (pas persistant)
_geocode_cache: Dict[str, dict] = {}
_geolocator = Nominatim(user_agent="rag_poc_geo_tools")


def get_gps_coordinates(city_name: str) -> dict:
    """
    Récupère les coordonnées GPS d'une ville via Nominatim.

    Returns:
      {"ok": bool, "city": str, "lat": float|None, "lon": float|None, "error": str|None}
    """
    try:
        key = (city_name or "").strip().lower()
        if not key:
            return {
                "ok": False,
                "city": city_name,
                "lat": None,
                "lon": None,
                "error": "ville vide",
            }

        if key in _geocode_cache:
            return _geocode_cache[key]

        # Respect Nominatim usage policy (prudence)
        time.sleep(1)
        location = _geolocator.geocode(city_name)

        if not location:
            res = {
                "ok": False,
                "city": city_name,
                "lat": None,
                "lon": None,
                "error": "ville non trouvée",
            }
            _geocode_cache[key] = res
            return res

        res = {
            "ok": True,
            "city": city_name,
            "lat": float(location.latitude),
            "lon": float(location.longitude),
            "error": None,
        }
        _geocode_cache[key] = res
        return res

    except Exception as e:
        return {
            "ok": False,
            "city": city_name,
            "lat": None,
            "lon": None,
            "error": str(e),
        }


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> dict:
    """
    Calcule la distance à vol d'oiseau (km) via geodesic.

    Returns:
      {"ok": bool, "distance_km": float|None, "error": str|None}
    """
    try:
        d = geodesic((lat1, lon1), (lat2, lon2)).kilometers
        return {"ok": True, "distance_km": round(float(d), 2), "error": None}
    except Exception as e:
        return {"ok": False, "distance_km": None, "error": str(e)}


def calculate_flight_time(distance_km: float, speed_kmh: float) -> dict:
    """
    Temps de vol en heures (arrondi à 2 décimales).

    Returns:
      {"ok": bool, "hours": float, "error": str|None}
    """
    try:
        if speed_kmh <= 0:
            return {"ok": False, "hours": 0.0, "error": "vitesse invalide"}
        return {
            "ok": True,
            "hours": round(float(distance_km) / float(speed_kmh), 2),
            "error": None,
        }
    except Exception as e:
        return {"ok": False, "hours": 0.0, "error": str(e)}
