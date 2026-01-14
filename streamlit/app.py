import streamlit as st
import atexit
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Imports normaux
from rag_agent_init import init_state
from rag_agent_dialog import answer_once
from tools.llama_server import LlamaServer

st.set_page_config(page_title="Assistant Vol d'Oiseau", page_icon="🦅")


@st.cache_resource
def init_app():
    """Init globale avec démarrage Llama"""
    with st.spinner("⚙️ Initialisation du modèle (20-30s)..."):
        state = init_state()

        if state.settings.use_llm:
            from tools.llama_server import LlamaServer

            server = LlamaServer(
                server_bin=state.settings.llama_server_bin,
                model_path=state.settings.llama_model_path,
                host=state.settings.llama_server_host,
                port=state.settings.llama_server_port,
                ctx_size=state.settings.llama_server_ctx_size,
                n_gpu_layers=state.settings.llama_server_n_gpu_layers,
            )
            server.start(timeout_s=30)
            state.llama_server = server
            atexit.register(server.stop)

    return state


state = init_app()

st.title("🦅 Assistant Vol d'Oiseau")
st.caption("✨ Calculs de distances et informations sur les oiseaux")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Distance entre deux villes ? Info sur un oiseau ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyse en cours..."):
            response = answer_once(state, prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
