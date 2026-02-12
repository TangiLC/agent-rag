from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BedrockSettings:
    region: str
    model_id: str
    max_tokens: int = 300
    temperature: float = 0.2
    top_p: float = 0.9


class BedrockQwenClient:
    def __init__(self, settings: BedrockSettings):
        self.settings = settings
        self._client = boto3.client("bedrock-runtime", region_name=settings.region)

    def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        mt = int(max_tokens or self.settings.max_tokens)
        temp = float(temperature if temperature is not None else self.settings.temperature)
        p = float(top_p if top_p is not None else self.settings.top_p)

        try:
            res = self._client.converse(
                modelId=self.settings.model_id,
                system=[{"text": system}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user}],
                    }
                ],
                inferenceConfig={
                    "maxTokens": mt,
                    "temperature": temp,
                    "topP": p,
                },
            )
        except (ClientError, BotoCoreError):
            logger.exception("Bedrock converse failed")
            raise

        try:
            content = res["output"]["message"]["content"]
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            return "\n".join(t for t in texts if t).strip()
        except Exception as exc:
            raise RuntimeError(f"Unexpected Bedrock response format: {exc}") from exc

    def chat_json(self, *, system: str, user: str, max_tokens: int = 300) -> Dict[str, Any]:
        txt = self.chat(system=system, user=user, max_tokens=max_tokens)
        txt = txt.strip()
        if txt.startswith("```json"):
            txt = txt[7:]
        elif txt.startswith("```"):
            txt = txt[3:]
        if txt.endswith("```"):
            txt = txt[:-3]
        txt = txt.strip()
        l = txt.find("{")
        r = txt.rfind("}")
        if l >= 0 and r > l:
            txt = txt[l : r + 1]
        return json.loads(txt)
