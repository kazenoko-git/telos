"""
OpenAI-Compatible REST API Adapter.

Zero-dependency HTTP client compatible with:
- vLLM / SGLang / TGI / Ollama local servers
- AFM 3 Core endpoints
- OpenAI / DeepSeek / Mistral / Anthropic proxies
- Concurrent multi-threaded evaluation via ThreadPoolExecutor
"""

import os
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from .base import BaseModelAdapter


class OpenAIAPIAdapter(BaseModelAdapter):
    """Adapter for any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        concurrency: int = 8,
        use_chat_endpoint: bool = True,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,
    ):
        super().__init__(model_name=model_name, backend_name="openai_api")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.concurrency = max(1, concurrency)
        self.use_chat_endpoint = use_chat_endpoint
        self.system_prompt = system_prompt
        self.timeout = timeout

    def _make_http_request(self, endpoint: str, payload: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
        """Performs a robust HTTP POST with exponential backoff retry on 429/5xx."""
        url = f"{self.api_base}{endpoint}"
        body_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Telos-Universal-Eval/2.0",
        }

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")

        delay = 1.0
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                status = err.code
                error_body = err.read().decode("utf-8", errors="replace")
                # Retry on rate limits (429) or transient server errors (500, 502, 503, 504)
                if status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise RuntimeError(
                    f"OpenAI API Error {status} from {url}: {error_body}"
                ) from err
            except (urllib.error.URLError, TimeoutError) as net_err:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise RuntimeError(f"Network error connecting to {url}: {net_err}") from net_err

        raise RuntimeError(f"Failed to receive response from {url} after {max_retries} attempts.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generates completion via /chat/completions or /completions."""
        if self.use_chat_endpoint:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }
            if stop:
                payload["stop"] = stop

            resp = self._make_http_request("/chat/completions", payload)
            choices = resp.get("choices", [])
            if not choices:
                return ""
            msg = choices[0].get("message", {})
            return msg.get("content") or ""
        else:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }
            if stop:
                payload["stop"] = stop

            resp = self._make_http_request("/completions", payload)
            choices = resp.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("text") or ""

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """Dispatches batch generation across worker threads for high throughput."""
        if self.concurrency <= 1 or len(prompts) <= 1:
            return [
                self.generate(p, max_new_tokens=max_new_tokens, temperature=temperature, stop=stop, **kwargs)
                for p in prompts
            ]

        def _worker(p):
            return self.generate(p, max_new_tokens=max_new_tokens, temperature=temperature, stop=stop, **kwargs)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(_worker, prompts))
