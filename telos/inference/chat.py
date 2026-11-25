"""Chat message formatting and Jinja2 template rendering for native inference."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


def _token_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return ""


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                raise ValueError("Unsupported multimodal content part; only text parts are supported.")
        return "".join(parts)
    raise ValueError("Message content must be a string, null, or list of text parts.")


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize OpenAI-format chat messages."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list.")
    normalized = []
    for entry in messages:
        if not isinstance(entry, dict):
            raise ValueError("Each message must be a JSON object.")
        role = entry.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("Each message must include a non-empty string role.")
        item = dict(entry)
        item["role"] = role.strip()
        item["content"] = _normalize_content(entry.get("content"))
        normalized.append(item)
    return normalized


@dataclass(frozen=True)
class ChatTemplateRenderer:
    """Render chat messages using a checkpoint's Jinja2 chat template or fallback."""

    template: str | None = None
    bos_token: str = ""
    eos_token: str = ""

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: str | Path | None, *, override: str | Path | None = None) -> "ChatTemplateRenderer":
        template = None
        bos_token = ""
        eos_token = ""
        if checkpoint_dir is not None:
            root = Path(checkpoint_dir)
            cfg_path = root / "tokenizer_config.json"
            if cfg_path.is_file():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid tokenizer_config.json: {cfg_path}") from exc
                if isinstance(cfg, dict):
                    raw_template = cfg.get("chat_template")
                    if isinstance(raw_template, str) and raw_template.strip():
                        template = raw_template
                    elif isinstance(raw_template, list):
                        for candidate in raw_template:
                            if isinstance(candidate, dict) and candidate.get("name") == "default" and isinstance(candidate.get("template"), str):
                                template = candidate["template"]
                                break
                    bos_token = _token_string(cfg.get("bos_token"))
                    eos_token = _token_string(cfg.get("eos_token"))
            jinja_path = root / "chat_template.jinja"
            if template is None and jinja_path.is_file():
                template = jinja_path.read_text(encoding="utf-8")

        if override is not None:
            override_path = Path(override)
            if override_path.is_file():
                template = override_path.read_text(encoding="utf-8")
            else:
                template = str(override)

        return cls(template=template, bos_token=bos_token, eos_token=eos_token)

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str:
        normalized = normalize_messages(messages)
        if not self.template:
            return _fallback_render(normalized, add_generation_prompt=add_generation_prompt)

        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError as exc:
            raise ImportError(
                "Rendering a checkpoint chat template requires Jinja2. "
                "Install it via `pip install 'telos-ml[serve]'`."
            ) from exc

        def _raise_exception(message: str):
            raise ValueError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = _raise_exception
        compiled = env.from_string(self.template)
        context: dict[str, Any] = {
            "messages": normalized,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        if enable_thinking is not None:
            context["enable_thinking"] = enable_thinking
        return compiled.render(**context)


def _fallback_render(messages: list[dict[str, Any]], *, add_generation_prompt: bool = True) -> str:
    """Simple role-prefixed fallback for base checkpoints without a chat template."""
    lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        lines.append(f"{role}: {content}")
    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n\n".join(lines)
