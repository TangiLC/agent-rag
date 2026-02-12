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
from typing import Optional
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
        self.thread: Optional[threading.Thread] = None
        self._last_msg: Optional[str] = None

    def _display_loop(self) -> None:
        # évite un flash si l'opération est très courte
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

    def start(self) -> None:
        self.stop_event.clear()
        self._last_msg = None
        self.thread = threading.Thread(target=self._display_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=0.5)


class LlamaServer:
    """
    Wrapper minimal autour de llama.cpp 'llama-server' (API type OpenAI /v1/*).

    Objectifs:
    - stabilité (threads plafonnés, timeouts raisonnables)
    - compatibilité Windows / Linux
    - arrêt fiable (kill process group sur posix, terminate sur Windows)
    """

    def __init__(
        self,
        server_bin: str,
        model_path: Path,
        host: str = "127.0.0.1",
        port: int = 8077,
        ctx_size: int = 1400,  # tokens LLM
        n_gpu_layers: int = 8,  # valeur sûre pour RTX 3050 4Go; sur CPU-only mettre 0
        n_threads: int = 4,  # levier majeur anti-OOM
        extra_args: Optional[list[str]] = None,
    ):
        self.server_bin = server_bin
        self.model_path = Path(model_path)
        self.host = host
        self.port = port
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.extra_args = extra_args or []

        self.proc: Optional[subprocess.Popen] = None
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _popen_kwargs(self) -> dict:
        """
        Lance un process group distinct:
        - POSIX: os.setsid + killpg
        - Windows: CREATE_NEW_PROCESS_GROUP (CTRL_BREAK_EVENT possible), sinon terminate()
        """
        kwargs: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.STDOUT,
            "text": True,
        }

        if os.name == "posix":
            kwargs["preexec_fn"] = os.setsid
        else:
            # Windows
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        return kwargs

    def start(self, timeout_s: float = 30.0) -> None:
        # Déjà démarré
        if self.proc and self.proc.poll() is None:
            return

        # Si un ancien process existe mais est mort, on nettoie.
        if self.proc and self.proc.poll() is not None:
            self.stop()

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
            "-t",
            str(self.n_threads),
        ]
        if self.extra_args:
            cmd.extend(self.extra_args)

        # Petit garde-fou: plafonne certains runtimes qui aiment sur-paralléliser
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", str(self.n_threads))
        env.setdefault("MKL_NUM_THREADS", str(self.n_threads))

        self.proc = subprocess.Popen(cmd, env=env, **self._popen_kwargs())
        atexit.register(self.stop)

        deadline = time.time() + timeout_s
        last_err: Optional[Exception] = None

        while time.time() < deadline:
            # crash early
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server s'est arrêté pendant le démarrage.")

            try:
                r = self._session.get(f"{self.base_url}/v1/models", timeout=0.7)
                if r.status_code == 200:
                    return
            except Exception as e:
                last_err = e

            time.sleep(0.2)

        # Timeout: on arrête pour éviter un process zombie
        self.stop()
        if last_err:
            raise TimeoutError(
                f"llama-server ne répond pas (dernier souci: {type(last_err).__name__})."
            )
        raise TimeoutError("llama-server ne répond pas.")

    def stop(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return

        try:
            if os.name == "posix":
                # kill process group
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            else:
                # Windows: tentative de terminer le process group, sinon terminate
                try:
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                except Exception:
                    self.proc.terminate()
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass

        # attend un peu, puis kill si nécessaire
        try:
            self.proc.wait(timeout=2.0)
        except Exception:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                else:
                    self.proc.kill()
            except Exception:
                pass

    def _get_model_id(self, timeout_s: float = 1.0) -> str:
        r = self._session.get(f"{self.base_url}/v1/models", timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        # format OpenAI-like: {"data":[{"id":"..."}]}
        return data["data"][0]["id"]

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 160,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        timeout_s: int = 60,
    ) -> str:
        model_id = getattr(self, "_cached_model_id", None)  # optional cache
        if not model_id:
            model_id = self._get_model_id()
            self._cached_model_id = model_id

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }

        r = self._session.post(
            f"{self.base_url}/v1/chat/completions", json=payload, timeout=timeout_s
        )
        if r.status_code == 400 and "repeat_penalty" in payload:
            # fallback: certaines builds refusent repeat_penalty
            payload.pop("repeat_penalty", None)
            r = self._session.post(
                f"{self.base_url}/v1/chat/completions", json=payload, timeout=timeout_s
            )
        if r.status_code >= 400:
            raise RuntimeError(f"llama-server HTTP {r.status_code}: {r.text}")

        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def chat_with_feedback(
        self,
        system: str,
        user: str,
        show_waiting: bool = True,
        max_tokens: int = 160,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        timeout_s: int = 60,
    ) -> str:
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
            import config  # type: ignore

            interval = getattr(config, "WAIT_MESSAGE_INTERVAL", 2.0)
        except Exception:
            interval = 2.0

        messenger = WaitingMessenger(interval_s=float(interval))
        messenger.start()
        try:
            return self.chat(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                timeout_s=timeout_s,
            )
        finally:
            messenger.stop()
