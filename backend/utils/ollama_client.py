import httpx
import json
from typing import Optional
from utils.logger import logger
from config import settings


class OllamaClient:
    """Async client for local Ollama LLM server."""

    def __init__(self, base_url: str = None, model: str = None, timeout: int = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout
        self._available: Optional[bool] = None

    async def check_availability(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    async def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        """Generate text using Ollama. Returns empty string if unavailable."""
        if self._available is False:
            logger.warning("Ollama not available — returning empty response")
            return ""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 512},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            return ""
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code}")
            return ""
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""

    async def generate_batch(self, prompts: list[str], system: str = "") -> list[str]:
        """Generate text for multiple prompts sequentially."""
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, system=system)
            results.append(result)
        return results


ollama = OllamaClient()
