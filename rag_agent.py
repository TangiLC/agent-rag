# rag_agent.py
from __future__ import annotations

import logging

from rag_agent_init import init_state
from rag_agent_dialog import repl


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:INFO|%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    setup_logging()
    state = init_state()
    repl(state)


if __name__ == "__main__":
    main()
