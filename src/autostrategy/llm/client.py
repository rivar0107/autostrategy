"""Unified LLM client for autostrategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autostrategy.config import LLMConfig, get_llm_api_key_status, resolve_llm_api_key


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str
    content: str


class LLMClient:
    """Client for calling LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = self._resolve_api_key()

    def _resolve_api_key(self) -> str | None:
        """Resolve API key from environment or known provider variables."""
        return resolve_llm_api_key(self.config)

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Send a chat request and return the content string."""
        if self.api_key is None:
            from autostrategy.services.exceptions import LLMConfigurationRequiredError

            raise LLMConfigurationRequiredError(
                get_llm_api_key_status(self.config),
                provider=self.config.provider,
            )
        if self.config.provider == "openai" or self.config.base_url:
            return self._chat_openai_compatible(messages, **kwargs)
        raise NotImplementedError(f"Provider '{self.config.provider}' is not supported yet.")

    def _chat_openai_compatible(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Call an OpenAI-compatible API."""
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for LLM calls. "
                "Install it with: pip install openai"
            ) from exc

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.config.base_url,
        )
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": message.role, "content": message.content} for message in messages],
            temperature=kwargs.get("temperature", self.config.temperature),
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty content.")
        return content
