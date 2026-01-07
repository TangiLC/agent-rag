from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class SourceRef:
    doc_id: str
    page: int
    chunk_id: str


def build_rag_prompt(question: str, contexts: List[str]) -> str:
    # Prompt volontairement strict : le modèle n'a "le droit" d'utiliser que CONTEXTE.
    joined = "\n\n---\n\n".join(contexts)
    return (
        "SYSTEM:\n"
        "Tu es un assistant RAG. Règles strictes :\n"
        "1) Utilise uniquement les informations du CONTEXTE.\n"
        "2) Si l'information n'est pas dans le CONTEXTE, dis : \"Je ne sais pas d'après le corpus.\".\n"
        "3) Réponse courte.\n\n"
        "CONTEXTE:\n"
        f"{joined}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "RÉPONSE:\n"
    )


def run_llama_cli(
    *,
    llama_cli_path: str,
    model_path: Path,
    prompt: str,
    n_predict: int,
    ctx_size: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    threads: int | None,
    timeout_s: int,
) -> str:
    cmd: List[str] = [
        llama_cli_path,
        "-m",
        str(model_path),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-c",
        str(ctx_size),
        "--temp",
        str(temperature),
        "--top-p",
        str(top_p),
        "--repeat-penalty",
        str(repeat_penalty),
        "--no-display-prompt",
    ]
    if threads is not None and threads > 0:
        cmd += ["-t", str(threads)]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    # llama.cpp écrit souvent sur stdout; en cas d'erreur, stderr est utile
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        raise RuntimeError(
            f"llama-cli a échoué (code={proc.returncode}). stderr: {err[:800]}"
        )

    return (proc.stdout or "").strip()


def format_sources(sources: Iterable[SourceRef]) -> str:
    lines = ["Sources:"]
    for s in sources:
        lines.append(f"- {s.doc_id} / p.{s.page} / {s.chunk_id}")
    return "\n".join(lines)
