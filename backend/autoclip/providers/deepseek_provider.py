"""DeepSeek provider, through DeepSeek's Anthropic-compatible API.

DeepSeek serves the Anthropic Messages API, so this is the Anthropic provider
with a different endpoint, model, and key. The request code is inherited
unchanged: no temperature and no assistant prefill.
"""

from __future__ import annotations

from .anthropic_provider import AnthropicProvider

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic"

SUGGESTED_MODELS = [DEFAULT_MODEL]


class DeepSeekProvider(AnthropicProvider):
    name = "deepseek"
    requires_key = True

    display_name = "DeepSeek"
    default_model = DEFAULT_MODEL
    #: Always sent explicitly, so ANTHROPIC_BASE_URL in the environment can
    #: never redirect DeepSeek traffic (or its key) somewhere else.
    default_base_url = DEFAULT_BASE_URL
    suggested_models = SUGGESTED_MODELS
    billing_hint = "Check your balance at platform.deepseek.com."
