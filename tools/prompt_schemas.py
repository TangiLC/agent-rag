# tools/prompt_schemas.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BirdRequest:
    """Représentation d'une demande concernant un oiseau"""

    request: bool  # True si l'utilisateur mentionne explicitement un oiseau
    name: Optional[str]  # Nom de l'oiseau si request=True, sinon None

    def has_specific_bird(self) -> bool:
        """Retourne True si un oiseau spécifique est demandé"""
        return self.request and self.name is not None

    def get_bird_or_fallback(self, fallback: str = "goéland") -> str:
        """Retourne le nom de l'oiseau ou le fallback si non spécifié"""
        if self.has_specific_bird():
            return self.name.lower()
        return fallback


@dataclass(frozen=True)
class InterpretedQuery:
    """Représentation structurée d'une requête utilisateur après parsing LLM"""

    ville_a: Optional[str]
    ville_b: Optional[str]
    bird: BirdRequest

    def has_cities(self) -> bool:
        """Retourne True si deux villes sont présentes"""
        return self.ville_a is not None and self.ville_b is not None

    def is_valid(self) -> bool:
        """
        Retourne True si la requête contient au moins des villes OU un oiseau.
        Sinon, on doit répondre "pas d'info à ce sujet".
        """
        return self.has_cities() or self.bird.request

    def needs_geo_computation(self) -> bool:
        """Retourne True si on doit calculer distance et durée"""
        return self.has_cities()

    def get_bird_name(self, fallback: str = "goéland") -> str:
        """Retourne le nom de l'oiseau à utiliser (demandé ou fallback)"""
        return self.bird.get_bird_or_fallback(fallback)


def parse_interpreted_query(json_data: dict) -> InterpretedQuery:
    """
    Construit un InterpretedQuery depuis un dict JSON.

    Args:
        json_data: Dict issu du parsing JSON du LLM

    Returns:
        InterpretedQuery validé

    Raises:
        KeyError, TypeError: si le format JSON est invalide
    """
    # Extraction des villes (peuvent être null)
    ville_a = json_data.get("ville_a")
    ville_b = json_data.get("ville_b")

    # Extraction de bird (obligatoire avec structure)
    bird_data = json_data.get("bird", {})
    if not isinstance(bird_data, dict):
        raise TypeError("Le champ 'bird' doit être un objet JSON")

    bird_request = bool(bird_data.get("request", False))
    bird_name = bird_data.get("name")

    # Normalisation du nom d'oiseau
    if bird_name and isinstance(bird_name, str):
        bird_name = bird_name.strip().lower()
        if not bird_name:  # Si vide après strip
            bird_name = None

    bird = BirdRequest(request=bird_request, name=bird_name)

    return InterpretedQuery(ville_a=ville_a, ville_b=ville_b, bird=bird)
