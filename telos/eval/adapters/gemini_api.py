"""
Google Gemini API REST Adapter.

Zero-dependency HTTP client for evaluating Google Gemini models:
- Gemini 4 26B A4B
- Gemini 1.5 Pro / Flash
- Compatible with official Google Generative Language API
"""

import os
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from .base import BaseModelAdapter


class GeminiAPIAdapter(BaseModelAdapter):
    """Adapter for Google Generative Language API (Gemini)."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        concurrency: int = 4,
        timeout: float = 60.0,
    ):
        super().__init__(model_name=model_name, backend_name="gemini_api")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            print("Warning: GEMINI_API_KEY environment variable not set.")
        self.concurrency = max(1, concurrency)
        self.timeout = timeout

    def _call_gemini_api(self, prompt: str, max_new_tokens: int, temperature: float, stop: Optional[List[str]]) -> str:
        """Sends a generateContent request to Google Gemini API."""
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        gen_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_new_tokens,
        }
        if stop:
            gen_config["stopSequences"] = stop

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": gen_config,
        }

        body_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Telos-Universal-Eval/2.0",
        }

        req = urllib.request.Request(endpoint, data=body_bytes, headers=headers, method="POST")

        max_retries = 5
        delay = 1.0
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return ""
                    parts = candidates[0].get("content", {}).get("parts", [])
                    non_thought = [p.get("text", "") for p in parts if "text" in p and not p.get("thought", False)]
                    if non_thought:
                        return "".join(non_thought).strip()
                    text_pieces = [p.get("text", "") for p in parts if "text" in p]
                    return "".join(text_pieces).strip()
            except urllib.error.HTTPError as err:
                status = err.code
                error_body = err.read().decode("utf-8", errors="replace")
                if status in (429, 500, 503) and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise RuntimeError(
                    f"Gemini API Error {status}: {error_body}"
                ) from err
            except (urllib.error.URLError, TimeoutError) as net_err:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise RuntimeError(f"Network error connecting to Gemini API: {net_err}") from net_err

        return ""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generates completion via Google Gemini REST endpoint."""
        return self._call_gemini_api(prompt, max_new_tokens, temperature, stop)

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """Dispatches multiple concurrent requests to Google Gemini API."""
        if self.concurrency <= 1 or len(prompts) <= 1:
            return [self.generate(p, max_new_tokens, temperature, stop, **kwargs) for p in prompts]

        def _worker(p):
            return self.generate(p, max_new_tokens, temperature, stop, **kwargs)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(_worker, prompts))
