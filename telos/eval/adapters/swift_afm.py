"""
Swift AFM Adapter
Evaluation adapter over telos.afm for Apple on-device Foundation Models.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from telos.afm import AFMBridge, AFMUnavailableError
from telos.afm.availability import probe

from .base import BaseModelAdapter


class SwiftAFMAdapter(BaseModelAdapter):
    """Adapter for Apple Foundation Models accessed through the Swift bridge."""

    def __init__(self, model_identifier: str = "afm-3-core-advanced", **kwargs):
        super().__init__(model_name=model_identifier, backend_name="swift_afm")
        self._bridge: Optional[AFMBridge] = None

    @property
    def bridge(self) -> AFMBridge:
        """Lazily starts the bridge so constructing the adapter stays cheap."""
        if self._bridge is None:
            self._bridge = AFMBridge()
        return self._bridge

    def availability(self):
        """Returns the :class:`~telos.afm.availability.AFMAvailability` report."""
        return probe()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generates a completion, or "" with a warning if unavailable."""
        instructions = kwargs.get("instructions")
        try:
            return self.bridge.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop=stop,
                instructions=instructions,
            )
        except AFMUnavailableError as exc:
            print(f"  [AFM Swift Bridge Warning] {exc}", file=sys.stderr)
            return ""

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """Sequential generation reusing the one warm bridge process."""
        return [
            self.generate(
                p,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop=stop,
                **kwargs
            )
            for p in prompts
        ]

    def close(self) -> None:
        """Terminates the bridge process."""
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None

    def __del__(self):
        self.close()
