#!/usr/bin/env python3
"""Unified model adapters mapping inference runtimes to the evaluation contract.

Supported runtimes:
- XERJ Reader-1 (API baseline via https://platform.xerj.ai/v1/answers)
- LM Studio local endpoints (http://localhost:1234/v1):
    - Ternary Bonsai 4 (ternary-bonsai-4b-mlx)
    - Ministral 3 8B Instruct 2512 (ministral-3-8b-instruct-2512)
    - Gemma 4 E4B (google/gemma-4-e4b)
- LFM3 Omni 2B (MLX direct via mlx-vlm persistent worker)
- Gemini 3.5 Flash-Lite (Google AI Studio / Generative Language API)
"""

import abc
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent

# Standard evaluation statuses accepted across comparative suite
VALID_STATUSES = {"ANSWER", "ABSENT", "UNKNOWN", "CONFLICT", "AMBIGUOUS", "MALFORMED", "ERROR"}

# OpenAI-compatible JSON Schema for LM Studio structured outputs with object support for money
LMSTUDIO_JSON_SCHEMA = {
    "name": "reader_evaluation_contract",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["ANSWER", "ABSENT", "UNKNOWN", "CONFLICT", "AMBIGUOUS"]
            },
            "value": {
                "type": ["string", "number", "boolean", "array", "object", "null"],
                "properties": {
                    "amount": {"type": ["string", "number"]},
                    "currency": {"type": "string"}
                }
            },
            "confidence": {"type": "number"},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quote": {"type": "string"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"}
                    },
                    "required": ["quote"]
                }
            },
            "reason": {"type": "string"}
        },
        "required": ["status", "confidence", "evidence", "reason"]
    }
}


def key_from_env(key_name: str) -> str:
    """Retrieve secret key from environment or project .env safely."""
    val = os.environ.get(key_name, "")
    if val:
        return val.strip().strip("\"'")
    env_paths = [REPO_ROOT / ".env", REPO_ROOT.parent / "telos" / ".env"]
    for path in env_paths:
        if path.exists():
            for line in path.read_text().splitlines():
                k, sep, v = line.partition("=")
                if sep and k.strip() in (key_name, f"export {key_name}"):
                    val = v.strip().strip("\"'")
                    if val:
                        return val
    return ""


def normalize_date(val: Any) -> Any:
    """Normalize various date formats into standard ISO 8601 YYYY-MM-DD."""
    if not isinstance(val, str):
        return val
    s = val.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    formats = [
        "%Y-%m-%d",
        "%d %B %Y",
        "%B %d, %Y",
        "%B %d %Y",
        "%d %b %Y",
        "%b %d, %Y",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%d-%m-%Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date().isoformat()  # noqa: DTZ007
        except ValueError:
            pass
    return s


def normalize_money(raw_val: Any) -> Any:
    """Normalize money representation to standard dict with amount and currency without defaulting to USD."""
    if isinstance(raw_val, dict):
        amount = raw_val.get("amount")
        currency = raw_val.get("currency")
        if amount is not None:
            cleaned_amount = (
                str(amount)
                .replace("$", "")
                .replace("€", "")
                .replace("£", "")
                .replace("¥", "")
                .replace(",", "")
                .strip()
            )
            cleaned_currency = str(currency).upper().strip() if currency else None
            return {"amount": cleaned_amount, "currency": cleaned_currency}
        return raw_val
    if isinstance(raw_val, str):
        s = raw_val.strip()
        m_code = re.search(r"\b([A-Za-z]{3})\b", s)
        currency = None
        if m_code:
            code = m_code.group(1).upper()
            if code in {"USD", "CAD", "EUR", "GBP", "JPY", "AUD", "CHF", "CNY", "INR", "NZD", "SEK"}:
                currency = code
        if not currency:
            if "€" in s:
                currency = "EUR"
            elif "£" in s:
                currency = "GBP"
            elif "¥" in s:
                currency = "JPY"
            elif "$" in s:
                if "C$" in s or "CAD" in s:
                    currency = "CAD"
                elif "A$" in s or "AUD" in s:
                    currency = "AUD"
                else:
                    currency = "USD"
        m = re.search(r"[-+]?\d*\.?\d+", s.replace(",", ""))
        if m:
            return {"amount": m.group(0), "currency": currency}
        return raw_val
    if isinstance(raw_val, (int, float, Decimal)):
        return {"amount": str(raw_val), "currency": None}
    return raw_val


def normalize_value(expected_type: str | None, raw_val: Any) -> Any:
    """Normalize extracted value to match expected data type and gold comparison."""
    if raw_val is None:
        return None

    if expected_type in ("money", "currency"):
        return normalize_money(raw_val)

    if expected_type == "date":
        return normalize_date(raw_val)

    if expected_type in ("pct", "percentage"):
        if isinstance(raw_val, (int, float)):
            return str(raw_val)
        if isinstance(raw_val, str):
            s = raw_val.strip().rstrip("%").strip()
            if "," in s and "." not in s:
                s = s.replace(",", "")
            try:
                d = Decimal(s)
                return str(d) if "." in s else str(int(d))
            except (InvalidOperation, ValueError):
                return s
        return raw_val

    if expected_type == "number":
        if isinstance(raw_val, (int, float)):
            return raw_val
        if isinstance(raw_val, str):
            s = raw_val.strip().rstrip("%").strip()
            if "," in s and "." not in s:
                s = s.replace(",", "")
            try:
                return float(s) if "." in s else int(s)
            except ValueError:
                return raw_val

    elif expected_type == "list":
        if isinstance(raw_val, list):
            return raw_val
        if isinstance(raw_val, str):
            try:
                parsed = json.loads(raw_val)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

    elif expected_type == "boolean":
        if isinstance(raw_val, bool):
            return raw_val
        if isinstance(raw_val, str):
            if raw_val.lower() in ("true", "yes", "1"):
                return True
            if raw_val.lower() in ("false", "no", "0"):
                return False

    return raw_val


def normalize_evidence(doc: str, raw_evidence: Any) -> list[dict[str, Any]]:
    """Normalize evidence quotes and calculate document character offsets."""
    if not raw_evidence:
        return []
    if isinstance(raw_evidence, str) or not isinstance(raw_evidence, list):
        raw_evidence = [raw_evidence]

    normalized: list[dict[str, Any]] = []
    for item in raw_evidence:
        quote = None
        start = None
        end = None
        if isinstance(item, str):
            quote = item
        elif isinstance(item, dict):
            quote = item.get("quote") or item.get("text")
            start = item.get("start")
            end = item.get("end")

        if not quote or not isinstance(quote, str):
            continue

        # Check if pre-existing offsets are strictly valid
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(doc)
            and doc[start:end] == quote
        ):
            normalized.append({"start": start, "end": end, "quote": quote})
            continue

        # Locate verbatim quote in document text
        idx = doc.find(quote)
        if idx != -1:
            normalized.append({
                "start": idx,
                "end": idx + len(quote),
                "quote": quote
            })
        else:
            # Keep original invalid offsets so evaluation grades provenance failure
            normalized.append({
                "start": start if isinstance(start, int) else -1,
                "end": end if isinstance(end, int) else -1,
                "quote": quote
            })
    return normalized


class BaseAdapter(abc.ABC):
    """Abstract model adapter contract."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abc.abstractmethod
    def query(self, case: dict[str, Any]) -> dict[str, Any]:
        """Query the model with a benchmark case and return normalized contract."""
        raise NotImplementedError

    def close(self) -> None:
        """Clean up background processes or persistent sessions."""


class XerjReaderAdapter(BaseAdapter):
    """Adapter for XERJ Reader-1 platform API baseline."""

    def __init__(self, api_key: str | None = None, base_url: str = "https://platform.xerj.ai"):
        super().__init__("xerj-reader-1")
        self.api_key = api_key or key_from_env("XERJ_API_KEY")
        self.base_url = base_url.rstrip("/")

    def query(self, case: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            return {
                "model": self.model_id,
                "status": "ERROR",
                "value": None,
                "value_text": None,
                "confidence": 0.0,
                "evidence": [],
                "reason": "XERJ_API_KEY is missing from environment or .env",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "latency_ms": 0.0,
                "client_ms": 0.0,
                "raw_response": {},
                "error": {"kind": "missing_api_key", "message": "XERJ_API_KEY missing"}
            }

        payload = {k: case[k] for k in ("document", "question", "type", "ref_date") if k in case}
        payload["model"] = self.model_id

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/answers",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "telos-evaluation-adapter/1.6.0"
            }
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = json.load(resp)
            client_ms = round((time.monotonic() - t0) * 1000, 2)
            usage = raw.get("usage", {})
            return {
                "model": self.model_id,
                "status": raw.get("status", "MALFORMED"),
                "value": raw.get("value"),
                "value_text": raw.get("value_text") or (str(raw.get("value")) if raw.get("value") is not None else None),
                "confidence": raw.get("confidence"),
                "evidence": normalize_evidence(case["document"], raw.get("evidence", [])),
                "reason": raw.get("reason"),
                "symbolic": raw.get("symbolic"),
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                "latency_ms": float(raw.get("latency_ms", client_ms)),
                "client_ms": client_ms,
                "raw_response": raw,
                "error": None
            }
        except urllib.error.HTTPError as err:
            client_ms = round((time.monotonic() - t0) * 1000, 2)
            body = err.read().decode("utf-8", errors="replace")[:2000]
            kind = "rate_limit_error" if err.code == 429 else "http_error"
            return {
                "model": self.model_id,
                "status": "ERROR",
                "value": None,
                "value_text": None,
                "confidence": 0.0,
                "evidence": [],
                "reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "latency_ms": 0.0,
                "client_ms": client_ms,
                "raw_response": {"http_status": err.code, "body": body},
                "error": {"kind": kind, "http_status": err.code, "body": body}
            }
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            client_ms = round((time.monotonic() - t0) * 1000, 2)
            return {
                "model": self.model_id,
                "status": "ERROR",
                "value": None,
                "value_text": None,
                "confidence": 0.0,
                "evidence": [],
                "reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "latency_ms": 0.0,
                "client_ms": client_ms,
                "raw_response": {},
                "error": {"kind": "transport_error", "message": str(err)}
            }


class LMStudioAdapter(BaseAdapter):
    """Adapter for LM Studio OpenAI-compatible local endpoints."""

    def __init__(self, model_id: str, base_url: str = "http://localhost:1234/v1"):
        super().__init__(model_id)
        self.base_url = base_url.rstrip("/")

    def _build_system_prompt(self) -> str:
        return (
            "You are a rigorous document question-answering assistant.\n"
            "Analyze the document and answer the question according to the requested type.\n"
            "Rules:\n"
            "1. status: 'ANSWER' if directly answerable; 'ABSENT' or 'UNKNOWN' if missing; "
            "'CONFLICT' or 'AMBIGUOUS' if contradictory or unresolved.\n"
            "2. value format by expected type:\n"
            "   - money: return object {\"amount\": \"15\", \"currency\": \"USD\"}\n"
            "   - pct: return percentage number without '%' (e.g. 10 or \"10\", not 0.1)\n"
            "   - date: return ISO 8601 date YYYY-MM-DD (e.g. 2024-02-27)\n"
            "   - number: return numeric value (e.g. 42 or 3.14)\n"
            "   - list: return JSON array of items (e.g. [\"a\", \"b\"])\n"
            "   - If status is not ANSWER, value must be null.\n"
            "3. evidence: List of verbatim quotes from the document proving your status and value.\n"
            "Respond strictly according to the json_schema."
        )

    def _build_user_prompt(self, case: dict[str, Any]) -> str:
        prompt = (
            f"Document:\n{case['document']}\n\n"
            f"Question: {case['question']}\n"
            f"Expected type: {case.get('type', 'string')}"
        )
        if "ref_date" in case:
            prompt += f"\nReference date: {case['ref_date']}"
        return prompt

    def query(self, case: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(case)}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": LMSTUDIO_JSON_SCHEMA
            },
            "temperature": 0.0,
            "max_tokens": 256
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        t0 = time.monotonic()
        raw = {}
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = json.load(resp)
            client_ms = round((time.monotonic() - t0) * 1000, 2)
            choice = raw.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            usage = raw.get("usage", {})

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as err:
                return {
                    "model": self.model_id,
                    "status": "MALFORMED",
                    "value": None,
                    "value_text": None,
                    "confidence": 0.0,
                    "evidence": [],
                    "reason": f"JSON decode failed: {err}",
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    },
                    "latency_ms": client_ms,
                    "client_ms": client_ms,
                    "raw_response": raw,
                    "error": {"kind": "malformed_response", "message": str(err), "raw_text": content}
                }

            if isinstance(parsed, list) or not isinstance(parsed, dict):
                parsed = {"status": "ANSWER", "value": parsed, "confidence": 1.0, "evidence": [], "reason": None}

            status = str(parsed.get("status", "ANSWER")).upper()
            if status not in VALID_STATUSES:
                status = "MALFORMED"

            val = normalize_value(case.get("type"), parsed.get("value"))
            evidence = normalize_evidence(case["document"], parsed.get("evidence", []))

            return {
                "model": self.model_id,
                "status": status,
                "value": val,
                "value_text": str(val) if val is not None else None,
                "confidence": float(parsed.get("confidence", 1.0)),
                "evidence": evidence,
                "reason": parsed.get("reason"),
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                "latency_ms": client_ms,
                "client_ms": client_ms,
                "raw_response": raw,
                "error": None
            }
        except urllib.error.HTTPError as err:
            client_ms = round((time.monotonic() - t0) * 1000, 2)
            body = err.read().decode("utf-8", errors="replace")[:2000]
            kind = "rate_limit_error" if err.code == 429 else "http_error"
            return {
                "model": self.model_id,
                "status": "ERROR",
                "value": None,
                "value_text": None,
                "confidence": 0.0,
                "evidence": [],
                "reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "latency_ms": 0.0,
                "client_ms": client_ms,
                "raw_response": {"http_status": err.code, "body": body},
                "error": {"kind": kind, "http_status": err.code, "body": body}
            }
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            client_ms = round((time.monotonic() - t0) * 1000, 2)
            return {
                "model": self.model_id,
                "status": "ERROR",
                "value": None,
                "value_text": None,
                "confidence": 0.0,
                "evidence": [],
                "reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "latency_ms": 0.0,
                "client_ms": client_ms,
                "raw_response": raw,
                "error": {"kind": "transport_error", "message": str(err)}
            }


class Lfm3OmniAdapter(BaseAdapter):
    """Adapter for LiquidAI LFM3-Omni-2B using persistent MLX worker subprocess."""

    def __init__(
        self,
        model_path: str = "/Users/ivansamuel/.lmstudio/models/LiquidAI/LFM3-Omni-2B-MLX-4bit",
        python_bin: str = "/Users/ivansamuel/Documents/Projects/testrepos/mlx-vlm/.venv/bin/python"
    ):
        super().__init__("lfm3-omni-2b")
        self.model_path = model_path
        self.python_bin = python_bin
        self._proc: subprocess.Popen | None = None

    def _ensure_worker(self) -> None:
        """Start persistent background worker subprocess if not running."""
        if self._proc is not None and self._proc.poll() is None:
            return

        cmd = [self.python_bin, str(Path(__file__).resolve()), "--worker", "--model-path", self.model_path]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        ready_line = self._proc.stdout.readline()
        try:
            status = json.loads(ready_line)
            if not status.get("ready"):
                raise RuntimeError(f"Worker handshake failed: {ready_line}")
        except Exception as e:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"Failed to start LFM3 worker: {e}\n{stderr}") from e

    def query(self, case: dict[str, Any]) -> dict[str, Any]:
        t0 = time.monotonic()
        raw = {}
        try:
            self._ensure_worker()
            req_data = {
                "document": case["document"],
                "question": case["question"],
                "type": case.get("type", "string"),
                "ref_date": case.get("ref_date")
            }
            self._proc.stdin.write(json.dumps(req_data) + "\n")
            self._proc.stdin.flush()

            resp_line = self._proc.stdout.readline()
            client_ms = round((time.monotonic() - t0) * 1000, 2)
            if not resp_line:
                return {
                    "model": self.model_id,
                    "status": "ERROR",
                    "value": None,
                    "value_text": None,
                    "confidence": 0.0,
                    "evidence": [],
                    "reason": "Worker process closed unexpectedly",
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "latency_ms": 0.0,
                    "client_ms": client_ms,
                    "raw_response": {},
                    "error": {"kind": "transport_error", "message": "Worker EOF"}
                }

            raw = json.loads(resp_line)
            if raw.get("error"):
                return {
                    "model": self.model_id,
                    "status": "ERROR",
                    "value": None,
                    "value_text": None,
                    "confidence": 0.0,
                    "evidence": [],
                    "reason": raw.get("error"),
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "latency_ms": 0.0,
                    "client_ms": client_ms,
                    "raw_response": raw,
                    "error": {"kind": "contract_error", "message": raw["error"]}
                }

            text = raw.get("text", "").strip()
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()

            gen_ms = float(raw.get("latency_ms", client_ms))
            in_tok = int(raw.get("prompt_tokens", 0))
            out_tok = int(raw.get("gen_tokens", 0))

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as err:
                return {
                    "model": self.model_id,
                    "status": "MALFORMED",
                    "value": None,
                    "value_text": None,
                    "confidence": 0.0,
                    "evidence": [],
                    "reason": f"JSON decode failed: {err}",
                    "usage": {
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "total_tokens": in_tok + out_tok
                    },
                    "latency_ms": gen_ms,
                    "client_ms": client_ms,
                    "raw_response": raw,
                    "error": {"kind": "malformed_response", "message": str(err), "raw_text": text}
                }

            # If decoded JSON is a list or scalar, wrap cleanly to avoid AttributeError
            if isinstance(parsed, list) or not isinstance(parsed, dict):
                parsed = {"status": "ANSWER", "value": parsed, "confidence": 1.0, "evidence": [], "reason": None}

            status = str(parsed.get("status", "ANSWER")).upper()
            if status not in VALID_STATUSES:
                status = "MALFORMED"

            val = normalize_value(case.get("type"), parsed.get("value"))
            evidence = normalize_evidence(case["document"], parsed.get("evidence", []))

            return {
                "model": self.model_id,
                "status": status,
                "value": val,
                "value_text": str(val) if val is not None else None,
                "confidence": float(parsed.get("confidence", 1.0)),
                "evidence": evidence,
                "reason": parsed.get("reason"),
                "usage": {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": in_tok + out_tok
                },
                "latency_ms": gen_ms,
                "client_ms": client_ms,
                "raw_response": raw,
                "error": None
            }
        except (TimeoutError, OSError, ValueError, KeyError, TypeError) as err:
            client_ms = round((time.monotonic() - t0) * 1000, 2)
            self.close()
            return {
                "model": self.model_id,
                "status": "ERROR",
                "value": None,
                "value_text": None,
                "confidence": 0.0,
                "evidence": [],
                "reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "latency_ms": 0.0,
                "client_ms": client_ms,
                "raw_response": raw,
                "error": {"kind": "contract_error" if isinstance(err, (TypeError, ValueError, KeyError)) else "transport_error", "message": str(err)}
            }

    def close(self) -> None:
        """Terminate persistent worker subprocess cleanly."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except (OSError, TimeoutError):
                self._proc.kill()
            finally:
                self._proc = None


class GeminiRateLimiter:
    """Thread-safe rate limiter pacing requests to stay below 15 RPM (4.5s interval)."""

    def __init__(self, interval: float = 4.5):
        self.lock = threading.Lock()
        self.next_start = 0.0
        self.interval = interval

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
            delay = start - now
        if delay > 0:
            time.sleep(delay)


GEMINI_RATE_LIMITER = GeminiRateLimiter(interval=4.5)


class GeminiFlashLiteAdapter(BaseAdapter):
    """Adapter for Google Gemini 3.5 Flash-Lite via Google Generative Language API."""

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-3.5-flash-lite"):
        super().__init__("gemini-3.5-flash-lite")
        self.api_key = api_key or key_from_env("GEMINI_API_KEY") or key_from_env("GOOGLE_API_KEY")
        self.endpoint_model = model_name

    def query(self, case: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            return {
                "model": self.model_id,
                "status": "ERROR",
                "value": None,
                "value_text": None,
                "confidence": 0.0,
                "evidence": [],
                "reason": "Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment or .env",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "latency_ms": 0.0,
                "client_ms": 0.0,
                "raw_response": {},
                "error": {
                    "kind": "missing_api_key",
                    "message": "Neither GEMINI_API_KEY nor GOOGLE_API_KEY configured"
                }
            }

        prompt = (
            "You are a rigorous document question-answering assistant.\n"
            "Analyze the document and answer according to the requested type.\n"
            "Rules:\n"
            "1. status: 'ANSWER' if directly answerable; 'ABSENT' or 'UNKNOWN' if missing; "
            "'CONFLICT' or 'AMBIGUOUS' if contradictory or unresolved.\n"
            "2. value format by expected type:\n"
            "   - money: return object {\"amount\": \"15\", \"currency\": \"USD\"}\n"
            "   - pct: return percentage number without '%' (e.g. 10 or \"10\", not 0.1)\n"
            "   - date: return ISO 8601 date YYYY-MM-DD (e.g. 2024-02-27)\n"
            "   - number: return numeric value (e.g. 42 or 3.14)\n"
            "   - list: return JSON array of items (e.g. [\"a\", \"b\"])\n"
            "   - If status is not ANSWER, value must be null.\n"
            "3. evidence: List of verbatim quotes from the document proving status and value.\n\n"
            f"Document:\n{case['document']}\n\n"
            f"Question: {case['question']}\n"
            f"Expected type: {case.get('type', 'string')}\n"
        )
        if "ref_date" in case:
            prompt += f"Reference date: {case['ref_date']}\n"
        prompt += "Output strictly valid JSON with keys: status, value, confidence, evidence (list of {quote}), reason."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.endpoint_model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        max_retries = 5
        raw = {}

        for attempt in range(max_retries):
            GEMINI_RATE_LIMITER.wait()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = json.load(resp)
                client_ms = round((time.monotonic() - t0) * 1000, 2)
                cands = raw.get("candidates", [{}])[0]
                text = cands.get("content", {}).get("parts", [{}])[0].get("text", "")
                usage_meta = raw.get("usageMetadata", {})

                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as err:
                    return {
                        "model": self.model_id,
                        "status": "MALFORMED",
                        "value": None,
                        "value_text": None,
                        "confidence": 0.0,
                        "evidence": [],
                        "reason": f"JSON decode failed: {err}",
                        "usage": {
                            "input_tokens": usage_meta.get("promptTokenCount", 0),
                            "output_tokens": usage_meta.get("candidatesTokenCount", 0),
                            "total_tokens": usage_meta.get("promptTokenCount", 0) + usage_meta.get("candidatesTokenCount", 0)
                        },
                        "latency_ms": client_ms,
                        "client_ms": client_ms,
                        "raw_response": raw,
                        "error": {"kind": "malformed_response", "message": str(err), "raw_text": text}
                    }

                if isinstance(parsed, list) or not isinstance(parsed, dict):
                    parsed = {"status": "ANSWER", "value": parsed, "confidence": 1.0, "evidence": [], "reason": None}

                status = str(parsed.get("status", "ANSWER")).upper()
                if status not in VALID_STATUSES:
                    status = "MALFORMED"

                val = normalize_value(case.get("type"), parsed.get("value"))
                evidence = normalize_evidence(case["document"], parsed.get("evidence", []))

                in_tok = usage_meta.get("promptTokenCount", 0)
                out_tok = usage_meta.get("candidatesTokenCount", 0)

                return {
                    "model": self.model_id,
                    "status": status,
                    "value": val,
                    "value_text": str(val) if val is not None else None,
                    "confidence": float(parsed.get("confidence", 1.0)),
                    "evidence": evidence,
                    "reason": parsed.get("reason"),
                    "usage": {
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "total_tokens": in_tok + out_tok
                    },
                    "latency_ms": client_ms,
                    "client_ms": client_ms,
                    "raw_response": raw,
                    "error": None
                }
            except urllib.error.HTTPError as err:
                client_ms = round((time.monotonic() - t0) * 1000, 2)
                body = err.read().decode("utf-8", errors="replace")[:2000]
                if err.code in (429, 503) and attempt < max_retries - 1:
                    wait_s = 5.0 * (attempt + 1)
                    retry_after = err.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_s = max(float(retry_after), wait_s)
                        except ValueError:
                            pass
                    m = re.search(r"(\d+(\.\d+)?)s", body)
                    if m:
                        try:
                            parsed_s = float(m.group(1)) + 1.0
                            if parsed_s <= 30.0:
                                wait_s = max(parsed_s, wait_s)
                            else:
                                wait_s = 999.0
                        except ValueError:
                            pass
                    if wait_s <= 30.0:
                        time.sleep(wait_s)
                        continue

                kind = "rate_limit_error" if err.code == 429 else "http_error"
                return {
                    "model": self.model_id,
                    "status": "ERROR",
                    "value": None,
                    "value_text": None,
                    "confidence": 0.0,
                    "evidence": [],
                    "reason": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "latency_ms": 0.0,
                    "client_ms": client_ms,
                    "raw_response": {"http_status": err.code, "body": body},
                    "error": {"kind": kind, "http_status": err.code, "body": body}
                }
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                client_ms = round((time.monotonic() - t0) * 1000, 2)
                return {
                    "model": self.model_id,
                    "status": "ERROR",
                    "value": None,
                    "value_text": None,
                    "confidence": 0.0,
                    "evidence": [],
                    "reason": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "latency_ms": 0.0,
                    "client_ms": client_ms,
                    "raw_response": raw,
                    "error": {"kind": "transport_error", "message": str(err)}
                }


# Registry of supported model adapters
ADAPTER_FACTORIES = {
    "xerj-reader-1": lambda: XerjReaderAdapter(),
    "ternary-bonsai-4b-mlx": lambda: LMStudioAdapter("ternary-bonsai-4b-mlx"),
    "ministral-3-8b-instruct-2512": lambda: LMStudioAdapter("ministral-3-8b-instruct-2512"),
    "google/gemma-4-e4b": lambda: LMStudioAdapter("google/gemma-4-e4b"),
    "lfm3-omni-2b": lambda: Lfm3OmniAdapter(),
    "qwen3.5-0.8b-mlx": lambda: LMStudioAdapter("qwen3.5-0.8b-mlx"),
    "qwen3.5-2b-optiq": lambda: LMStudioAdapter("qwen3.5-2b-optiq"),
}


def get_adapter(model_id: str) -> BaseAdapter:
    """Factory creating an adapter instance for a supported model identifier."""
    if model_id not in ADAPTER_FACTORIES:
        raise ValueError(f"Unknown model identifier '{model_id}'. Supported: {list(ADAPTER_FACTORIES.keys())}")
    return ADAPTER_FACTORIES[model_id]()


def run_worker_loop(model_path: str) -> None:
    """Persistent MLX execution loop for LFM3-Omni-2B reading JSON lines over stdin."""
    try:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor = load(model_path)
    except Exception as e:  # noqa: BLE001
        sys.stdout.write(json.dumps({"ready": False, "error": str(e)}) + "\n")
        sys.stdout.flush()
        return

    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    system_prompt = (
        "You are a precise document question-answering assistant. Respond strictly with JSON.\n"
        "Format:\n"
        "{\n"
        '  "status": "ANSWER",\n'
        '  "value": ...,\n'
        '  "confidence": 1.0,\n'
        '  "evidence": [{"quote": "..."}],\n'
        '  "reason": "..."\n'
        "}\n"
        "Rules:\n"
        "- status: 'ANSWER' if answerable; 'ABSENT' or 'UNKNOWN' if missing; 'CONFLICT' or 'AMBIGUOUS' if contradictory.\n"
        "- value format by expected type:\n"
        '  - money: {"amount": "15", "currency": "USD"}\n'
        "  - pct: numeric percentage without % (e.g. 10)\n"
        "  - date: ISO YYYY-MM-DD\n"
        "  - list: array of items\n"
        "  - number: numeric value\n"
        "  - null if status is not ANSWER\n"
        "- evidence: list of verbatim quotes from document.\n"
        "Respond with text."
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            user_prompt = (
                f"Document:\n{req['document']}\n\n"
                f"Question: {req['question']}\n"
                f"Expected type: {req.get('type', 'string')}"
            )
            if req.get("ref_date"):
                user_prompt += f"\nReference date: {req['ref_date']}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt = apply_chat_template(processor, model.config, messages)
            t0 = time.monotonic()
            res = generate(model, processor, prompt, max_tokens=200, temperature=0.0)
            latency_ms = round((time.monotonic() - t0) * 1000, 2)

            out = {
                "text": res.text,
                "latency_ms": latency_ms,
                "prompt_tokens": getattr(res, "prompt_tokens", 0),
                "gen_tokens": getattr(res, "generation_tokens", 0)
            }
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
        except Exception as e:  # noqa: BLE001
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="Run as LFM3 persistent MLX worker")
    parser.add_argument("--model-path", default="/Users/ivansamuel/.lmstudio/models/LiquidAI/LFM3-Omni-2B-MLX-4bit")
    parser.add_argument("--test", choices=list(ADAPTER_FACTORIES.keys()), help="Test single adapter")
    parser.add_argument("--test-all", action="store_true", help="Test all adapters on a fixture case")
    args = parser.parse_args()

    if args.worker:
        run_worker_loop(args.model_path)
        return

    sample_case = {
        "id": "fixture-01",
        "category": "arithmetic",
        "document": "Diagnostic fixture. Alpha finished 14 tasks; Beta finished 5 tasks.",
        "question": "How many tasks did Alpha and Beta finish in total?",
        "type": "number",
        "expected": "19"
    }

    models_to_test = [args.test] if args.test else (list(ADAPTER_FACTORIES.keys()) if args.test_all else [])
    if not models_to_test:
        print(f"Supported models: {list(ADAPTER_FACTORIES.keys())}")
        print("Use --test <model_id> or --test-all to run diagnostic verification.")
        return

    for m in models_to_test:
        print(f"\n=== Testing adapter: {m} ===")
        adapter = get_adapter(m)
        try:
            res = adapter.query(sample_case)
            print(f"Status: {res['status']}")
            print(f"Value: {res['value']} (text: {res['value_text']})")
            print(f"Evidence: {res['evidence']}")
            print(f"Latency: {res['latency_ms']} ms (client: {res['client_ms']} ms)")
            print(f"Usage: {res['usage']}")
            if res.get("error"):
                print(f"Reported error: {res['error']}")
        finally:
            adapter.close()


if __name__ == "__main__":
    main()
