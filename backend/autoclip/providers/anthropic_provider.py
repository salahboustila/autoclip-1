"""Anthropic (Claude) provider."""

from __future__ import annotations

import logging

from .base import DetectionConfig, LLMProvider, ProviderError, ProviderStatus

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"

#: Shown in the settings UI. Kept short and current rather than exhaustive.
SUGGESTED_MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
]


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    requires_key = True

    #: Vendor details. A vendor serving an Anthropic-compatible API overrides
    #: these and inherits the request code unchanged (see deepseek_provider.py).
    display_name = "Anthropic"
    default_model = DEFAULT_MODEL
    #: None means the SDK's own default, api.anthropic.com.
    default_base_url: str | None = None
    suggested_models = SUGGESTED_MODELS
    billing_hint = "Check your plan at console.anthropic.com."

    def __init__(self, model: str = "", *, api_key: str | None = None, base_url: str | None = None):
        super().__init__(
            model or self.default_model,
            api_key=api_key,
            base_url=base_url or self.default_base_url,
        )

    def _client(self):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - anthropic is a core dep
            raise ProviderError(
                "The anthropic package is not installed.", provider=self.name
            ) from exc

        if not self.api_key:
            raise ProviderError(
                f"No {self.display_name} API key is set.",
                provider=self.name,
                hint=f"Add one with `autoclip config set-secret {self.name}`.",
            )

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncAnthropic(**kwargs)

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        client = self._client()
        try:
            # No temperature and no assistant prefill: current Claude models
            # (Sonnet 5, Opus 5, the 4.6+ family) reject both with a 400. Any
            # prose around the JSON is stripped by extract_json_object in base.py.
            message = await client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise _translate(exc, self) from exc

        return "".join(block.text for block in message.content if hasattr(block, "text"))

    async def health_check(self) -> ProviderStatus:
        if not self.api_key:
            return ProviderStatus(
                name=self.name,
                available=False,
                detail="No API key set",
                models=self.suggested_models,
            )
        try:
            client = self._client()
            await client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ok"}],
            )
        except Exception as exc:
            return ProviderStatus(
                name=self.name,
                available=False,
                detail=str(exc)[:200],
                models=self.suggested_models,
            )
        return ProviderStatus(
            name=self.name, available=True, detail=self.model, models=self.suggested_models
        )


def _translate(exc: Exception, provider: AnthropicProvider) -> ProviderError:
    message = str(exc)
    lowered = message.lower()
    vendor = provider.display_name

    if "authentication" in lowered or "invalid x-api-key" in lowered or "401" in lowered:
        return ProviderError(
            f"{vendor} rejected the API key.",
            provider=provider.name,
            hint=f"Re-add it with `autoclip config set-secret {provider.name}`.",
        )
    if "rate limit" in lowered or "429" in lowered:
        return ProviderError(
            f"{vendor} rate limit reached.",
            provider=provider.name,
            hint="Wait a moment and retry the job, or switch to a different provider.",
        )
    if "not_found" in lowered or "model" in lowered and "404" in lowered:
        return ProviderError(
            f"{vendor} does not recognise the model '{provider.model}'.",
            provider=provider.name,
            hint=f"Try one of: {', '.join(provider.suggested_models)}",
        )
    # DeepSeek reports an empty account as "Insufficient Balance" (HTTP 402).
    if "credit" in lowered or "billing" in lowered or "insufficient balance" in lowered:
        return ProviderError(
            f"{vendor} reports a billing or credit problem.",
            provider=provider.name,
            hint=provider.billing_hint,
        )
    return ProviderError(f"{vendor} request failed: {message}", provider=provider.name)
