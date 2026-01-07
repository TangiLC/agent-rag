from __future__ import annotations

import logging

from rag_agent_init import init_state
from rag_agent_dialog import repl
from tools.llama_server import LlamaServer


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:INFO|%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    setup_logging()
    state = init_state()

    if state.settings.use_llm:
        logging.info("Démarrage du serveur llama…")
        server = LlamaServer(
            server_bin=state.settings.llama_server_bin,
            model_path=state.settings.llama_model_path,
            host=state.settings.llama_server_host,
            port=state.settings.llama_server_port,
            ctx_size=state.settings.llama_server_ctx_size,
            n_gpu_layers=state.settings.llama_server_n_gpu_layers,
        )
        server.start(timeout_s=float(state.settings.llama_server_start_timeout_s))
        logging.info(f"Serveur llama prêt ({server.base_url})")
        state.llama_server = server
    else:
        logging.info("LLM désactivé (USE_LLM=False).")

    repl(state)


if __name__ == "__main__":
    main()
