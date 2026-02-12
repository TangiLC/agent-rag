from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools.bedrock_qwen import BedrockQwenClient, BedrockSettings
from tools.geo_tools import calculate_distance, calculate_flight_time, get_gps_coordinates

load_dotenv()

st.set_page_config(page_title="Assistant Vol d'Oiseau (Light)", page_icon=":bird:")


@dataclass(frozen=True)
class ParsedQuery:
    city_a: Optional[str]
    city_b: Optional[str]
    bird: Optional[str]


BIRD_SPEEDS_KMH = {
    "goeland": 40.0,
    "goeland brun": 40.0,
    "faucon pelerin": 130.0,
    "faucon": 110.0,
    "aigle royal": 80.0,
    "martinet noir": 111.0,
    "albatros": 70.0,
}


@st.cache_resource
def init_bedrock() -> BedrockQwenClient:
    region = os.getenv("AWS_REGION", "eu-west-1")
    model_id = os.getenv("BEDROCK_MODEL_ID", "qwen.qwen3-32b-instruct-v1:0")
    settings = BedrockSettings(
        region=region,
        model_id=model_id,
        max_tokens=int(os.getenv("BEDROCK_MAX_TOKENS", "300")),
        temperature=float(os.getenv("BEDROCK_TEMPERATURE", "0.2")),
        top_p=float(os.getenv("BEDROCK_TOP_P", "0.9")),
    )
    return BedrockQwenClient(settings=settings)


def _fallback_parse(question: str) -> ParsedQuery:
    q = " ".join((question or "").strip().split())
    city_a = None
    city_b = None

    m = re.split(r"\s*(?:->|→)\s*", q, maxsplit=1)
    if len(m) == 2 and m[0].strip() and m[1].strip():
        city_a, city_b = m[0].strip(" ,;:-"), m[1].strip(" ,;:-")

    if city_a is None or city_b is None:
        m = re.search(r"\bentre\s+(.+?)\s+et\s+(.+)", q, re.IGNORECASE)
        if m:
            city_a, city_b = m.group(1).strip(" ,;:-"), m.group(2).strip(" ,;:-")

    if city_a is None or city_b is None:
        m = re.search(r"\bde\s+(.+?)\s+(?:a|à)\s+(.+)", q, re.IGNORECASE)
        if m:
            city_a, city_b = m.group(1).strip(" ,;:-"), m.group(2).strip(" ,;:-")

    bird = None
    bird_match = re.search(
        r"\b(goeland|goéland|faucon(?:\s+pelerin)?|faucon(?:\s+pélerin)?|aigle(?:\s+royal)?|martinet(?:\s+noir)?|albatros)\b",
        q,
        re.IGNORECASE,
    )
    if bird_match:
        bird = bird_match.group(1).lower().replace("é", "e")

    return ParsedQuery(city_a=city_a, city_b=city_b, bird=bird)


def parse_query(client: BedrockQwenClient, question: str) -> ParsedQuery:
    system = (
        "Parse la requete utilisateur en JSON strict.\n"
        "Reponds uniquement en JSON: "
        '{"city_a": "X" ou null, "city_b": "Y" ou null, "bird": "nom" ou null}'
    )
    try:
        data = client.chat_json(system=system, user=question, max_tokens=120)
        return ParsedQuery(
            city_a=data.get("city_a"),
            city_b=data.get("city_b"),
            bird=data.get("bird"),
        )
    except Exception:
        return _fallback_parse(question)


def get_bird_info(client: BedrockQwenClient, bird: str) -> Tuple[Optional[float], str]:
    system = (
        "Tu reponds en JSON strict sur un oiseau.\n"
        'Format: {"speed_kmh": number ou null, "fact": "phrase courte"}'
    )
    try:
        data = client.chat_json(
            system=system,
            user=f"Donne une vitesse de vol typique et un fait bref sur: {bird}",
            max_tokens=140,
        )
        speed = data.get("speed_kmh")
        fact = str(data.get("fact") or "").strip()
        parsed_speed = float(speed) if speed is not None else None
        return parsed_speed, fact
    except Exception:
        return None, f"Oiseau de reference: {bird}"


def answer(client: BedrockQwenClient, question: str) -> str:
    parsed = parse_query(client, question)
    parts = []
    distance_km: Optional[float] = None

    if parsed.city_a and parsed.city_b:
        a = get_gps_coordinates(parsed.city_a)
        b = get_gps_coordinates(parsed.city_b)
        if a.get("ok") and b.get("ok"):
            d = calculate_distance(a["lat"], a["lon"], b["lat"], b["lon"])
            if d.get("ok"):
                distance_km = float(d["distance_km"])
                parts.append(
                    f"Distance (vol d'oiseau) {parsed.city_a} -> {parsed.city_b} : {distance_km} km."
                )

    bird = (parsed.bird or "goeland").strip().lower()
    speed = BIRD_SPEEDS_KMH.get(bird)
    llm_speed, fact = get_bird_info(client, bird)
    if speed is None:
        speed = llm_speed
    if fact:
        parts.append(fact)

    if distance_km is not None and speed and speed > 0:
        t = calculate_flight_time(distance_km, speed)
        if t.get("ok"):
            parts.append(f"Temps de vol estime (a {int(speed)} km/h) : {t['hours']} h.")

    if not parts:
        return "Je ne peux pas extraire d'information exploitable pour cette question."
    return "\n\n".join(parts)


st.title("Assistant Vol d'Oiseau - Light Deploy")
st.caption("Mode leger: geotools + Qwen sur Amazon Bedrock (sans embeddings locaux)")

client = init_bedrock()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Distance entre deux villes ? Info sur un oiseau ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            response = answer(client, prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
