# llama_server.py
import atexit
import os
import signal
import subprocess
import time
from pathlib import Path
import threading
import time
import random
import requests


class WaitingMessenger:
    """Affiche des messages d'attente aléatoires sans répétition immédiate."""

    VARIATIONS = ["", "encore ", "toujours "]

    MESSAGES = [
        "🤔 Je réfléchis {v}...",
        "📚 Je consulte {v}mes données...",
        "🔍 Recherche {v}en cours...",
        "⚙️ Analyse {v}en cours...",
        "💭 {v}en cours de réflexion...",
        "🧠 Ca titille {v}les neurones...",
        "💡 Si seulement la RAM était {v}moins chère...",
        "❓ La réponse est-elle {v}42 ?..",
        "❔ La réponse est {v}ailleurs !..",
        "🔮 Je consulte {v}les astres...",
        "📎 Une question a {v}été posée...",
        "🧩 Tout se met {v}en place...",
    ]

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = None
        self.start_time = None
        self._last_msg = None

    def _display_loop(self):
        if self.stop_event.wait(1.0):
            return

        while not self.stop_event.is_set():
            variation = random.choice(self.VARIATIONS)

            if self._last_msg is None or len(self.MESSAGES) == 1:
                template = random.choice(self.MESSAGES)
            else:
                template = random.choice(
                    [m for m in self.MESSAGES if m != self._last_msg]
                )

            msg = template.format(v=variation)
            self._last_msg = template

            print(f"\r{msg:<60}", end="", flush=True)

            if self.stop_event.wait(self.interval_s):
                break

        print("\r" + " " * 60 + "\r", end="", flush=True)

    def start(self):
        """Démarre l'affichage des messages."""
        self.start_time = time.time()
        self.stop_event.clear()
        self._last_msg = None
        self.thread = threading.Thread(target=self._display_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Arrête l'affichage des messages."""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=0.5)


class LlamaServer:
    def __init__(
        self,
        server_bin: str,
        model_path: Path,
        host="127.0.0.1",
        port=8077,
        ctx_size=2048,
        n_gpu_layers=999,
    ):
        self.server_bin = server_bin
        self.model_path = Path(model_path)
        self.host = host
        self.port = port
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.proc = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, timeout_s: float = 20.0) -> None:
        if self.proc and self.proc.poll() is None:
            return

        cmd = [
            self.server_bin,
            "-m",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-c",
            str(self.ctx_size),
            "-ngl",
            str(self.n_gpu_layers),
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        atexit.register(self.stop)

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server s'est arrêté au démarrage.")
            try:
                r = requests.get(f"{self.base_url}/v1/models", timeout=0.5)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.2)

        raise TimeoutError("llama-server ne répond pas.")

    def stop(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 200,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        timeout_s: int = 80,
    ) -> str:
        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
            # llama-server accepte généralement "repeat_penalty" en compatible llama.cpp ;
            # si ta build ne l'accepte pas, supprime ce champ.
            "repeat_penalty": repeat_penalty,
        }
        r = requests.post(
            f"{self.base_url}/v1/chat/completions", json=payload, timeout=timeout_s
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def chat_with_feedback(
        self,
        system: str,
        user: str,
        show_waiting: bool = True,
        max_tokens: int = 200,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        timeout_s: int = 80,
    ) -> str:
        """
        Version de chat() avec messages d'attente pendant la génération.

        Args:
            show_waiting: Si True, affiche des messages d'attente cycliques
            Autres args: identiques à chat()

        Returns:
            Réponse du LLM
        """
        if not show_waiting:
            return self.chat(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                timeout_s=timeout_s,
            )

        # Récupérer l'intervalle depuis config si disponible
        try:
            import config

            interval = getattr(config, "WAIT_MESSAGE_INTERVAL", 2.0)
        except (ImportError, AttributeError):
            interval = 2.0

        messenger = WaitingMessenger(interval_s=interval)
        messenger.start()

        try:
            result = self.chat(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                timeout_s=timeout_s,
            )
            return result
        finally:
            messenger.stop()
