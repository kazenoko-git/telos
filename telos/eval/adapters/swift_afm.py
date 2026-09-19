"""
Apple Foundation Models (AFM) Swift Bridge Adapter for Télos.

Communicates with the native Swift FoundationModels.framework binary via persistent IPC.
Provides access to on-device AFM 3 Core and AFM 3 Core Advanced on Apple Silicon.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .base import BaseModelAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWIFT_DIR = Path(__file__).resolve().parent / "swift_afm"
BINARY_PATH = SWIFT_DIR / "afm_bridge"
SOURCE_PATH = SWIFT_DIR / "afm_bridge.swift"


class SwiftAFMAdapter(BaseModelAdapter):
    """Adapter for Apple Foundation Models accessed through native Swift framework."""

    def __init__(self, model_identifier: str = "afm-3-core-advanced", **kwargs):
        super().__init__(model_name=model_identifier, backend_name="swift_afm")
        self._ensure_binary()
        self._proc = None
        self._start_process()

    def _ensure_binary(self):
        """Compiles the Swift bridge binary if not already present."""
        if BINARY_PATH.exists() and os.access(BINARY_PATH, os.X_OK):
            return

        SWIFT_DIR.mkdir(parents=True, exist_ok=True)
        if not SOURCE_PATH.exists():
            raise FileNotFoundError(f"Swift bridge source missing at {SOURCE_PATH}")

        print("Compiling native Swift AFM bridge with FoundationModels.framework...")
        cmd = [
            "swiftc",
            "-O",
            "-parse-as-library",
            str(SOURCE_PATH),
            "-o",
            str(BINARY_PATH),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to compile Swift AFM bridge: {res.stderr}")

    def _start_process(self):
        """Starts the persistent Swift bridge process."""
        self._proc = subprocess.Popen(
            [str(BINARY_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Sends generation request to persistent Swift AFM bridge."""
        if self._proc is None or self._proc.poll() is not None:
            self._start_process()

        instructions = kwargs.get(
            "instructions",
            "You are an expert programming and reasoning assistant. Provide direct, concise, and accurate output without conversational filler."
        )

        req = {
            "id": 1,
            "prompt": prompt,
            "instructions": instructions,
            "maxTokens": max_new_tokens,
            "temperature": temperature,
            "stopTokens": stop,
        }

        try:
            req_line = json.dumps(req) + "\n"
            self._proc.stdin.write(req_line)
            self._proc.stdin.flush()

            resp_line = self._proc.stdout.readline()
            if not resp_line:
                return ""

            data = json.loads(resp_line)
            if data.get("error"):
                print(f"  [AFM Swift Bridge Warning] {data['error']}", file=sys.stderr)
            return data.get("content", "")
        except Exception as e:
            print(f"  [AFM Swift Bridge Error] {e}", file=sys.stderr)
            return ""

    def close(self):
        """Terminates persistent bridge process."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __del__(self):
        self.close()
