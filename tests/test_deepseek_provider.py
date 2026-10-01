"""The DeepSeek provider: the Anthropic adapter pointed at DeepSeek's endpoint.

Everything runs against a fake ``AsyncAnthropic``, so no test touches the
network. The one test that builds the real SDK client only inspects it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest
from autoclip import app as app_module
from autoclip import config
from autoclip.app import create_app
from autoclip.db import store
from autoclip.db.models import Source, new_id
from autoclip.pipeline.runner import settings_for_job
from autoclip.providers import (
    PROVIDERS,
    AnthropicProvider,
    DeepSeekProvider,
    DetectionConfig,
    ProviderError,
    TranscriptWindow,
    build_provider,
)
from autoclip.providers.deepseek_provider import DEFAULT_BASE_URL, DEFAULT_MODEL
from fastapi.testclient import TestClient

DS_KEY = "sk-ds-TEST-NEVER-LOG-0001"
ANT_KEY = "sk-ant-TEST-0002"
VALID = '{"clips":[{"start_word_index":10,"end_word_index":50,"title":"T","score":80}]}'


def _text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


class FakeSDK:
    """Stands in for ``anthropic.AsyncAnthropic``.

    Records the arguments each client is built with and every request sent,
    and answers with queued replies (lists of content blocks) or ``error``.
    """

    def __init__(self) -> None:
        self.clients: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.replies: list[list[SimpleNamespace]] = []
        self.error: Exception | None = None

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.clients.append(kwargs)
        return SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        content = self.replies.pop(0) if self.replies else [_text('{"clips": []}')]
        return SimpleNamespace(content=content)


@pytest.fixture(autouse=True)
def _no_key_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    # An AUTOCLIP_*_KEY in the developer's environment beats the keyring.
    monkeypatch.delenv("AUTOCLIP_DEEPSEEK_KEY", raising=False)
    monkeypatch.delenv("AUTOCLIP_ANTHROPIC_KEY", raising=False)


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> FakeSDK:
    fake = FakeSDK()
    monkeypatch.setattr(anthropic, "AsyncAnthropic", fake)
    return fake


@pytest.fixture
def window() -> TranscriptWindow:
    return TranscriptWindow(text="[0]hello [1]world", first_word=0, last_word=100)


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


class TestRegistration:
    def test_registered_with_its_own_secret(self) -> None:
        assert PROVIDERS["deepseek"] is DeepSeekProvider
        assert DeepSeekProvider.requires_key is True
        assert "deepseek" in config.KEYED_PROVIDERS

    def test_reuses_the_anthropic_provider(self) -> None:
        assert issubclass(DeepSeekProvider, AnthropicProvider)

    def test_anthropic_stays_the_default(self, fake_keyring) -> None:
        settings = config.Settings()

        provider = build_provider(settings=settings)

        assert settings.active_provider == "anthropic"
        assert type(provider) is AnthropicProvider
        assert provider.base_url is None

    def test_defaults_when_unconfigured(self, fake_keyring) -> None:
        provider = build_provider("deepseek", config.Settings())

        assert provider.model == DEFAULT_MODEL == "deepseek-flash"
        assert provider.base_url == DEFAULT_BASE_URL == "https://api.deepseek.com/anthropic"

    def test_configured_model_and_base_url_win(self, fake_keyring) -> None:
        settings = config.Settings()
        settings.provider("deepseek").model = "deepseek-other"
        settings.provider("deepseek").base_url = "https://proxy.example/anthropic"

        provider = build_provider("deepseek", settings)

        assert provider.model == "deepseek-other"
        assert provider.base_url == "https://proxy.example/anthropic"


class TestRequest:
    async def test_sends_to_deepseek_with_the_deepseek_key(
        self, sdk: FakeSDK, fake_keyring
    ) -> None:
        config.set_secret("anthropic", ANT_KEY)
        config.set_secret("deepseek", DS_KEY)
        provider = build_provider("deepseek", config.load())

        await provider.complete("system text", "user text", DetectionConfig())

        assert sdk.clients == [{"api_key": DS_KEY, "base_url": DEFAULT_BASE_URL}]
        assert sdk.calls[0]["model"] == DEFAULT_MODEL
        assert sdk.calls[0]["system"] == "system text"

    async def test_no_temperature_and_no_prefill(
        self, sdk: FakeSDK, window: TranscriptWindow
    ) -> None:
        # The retry round must not add either one.
        sdk.replies = [[_text("not json")], [_text(VALID)]]
        provider = DeepSeekProvider(api_key=DS_KEY)

        result = await provider.detect_highlights(window, DetectionConfig(temperature=0.9))

        assert len(result.clips) == 1
        assert len(sdk.calls) == 2
        for call in sdk.calls:
            assert set(call) == {"model", "max_tokens", "system", "messages"}
            assert [m["role"] for m in call["messages"]] == ["user"]

    async def test_only_text_blocks_are_returned(self, sdk: FakeSDK) -> None:
        sdk.replies = [[SimpleNamespace(type="thinking", thinking="hmm"), _text(VALID)]]

        raw = await DeepSeekProvider(api_key=DS_KEY).complete("s", "u", DetectionConfig())

        assert raw == VALID


class TestHealthCheck:
    async def test_without_a_key_it_is_not_ready(self, sdk: FakeSDK) -> None:
        status = await DeepSeekProvider().health_check()

        assert status.name == "deepseek"
        assert status.available is False
        assert status.detail == "No API key set"
        assert status.models == [DEFAULT_MODEL]
        assert sdk.clients == []

    async def test_with_a_key_it_is_ready(self, sdk: FakeSDK) -> None:
        status = await DeepSeekProvider(api_key=DS_KEY).health_check()

        assert status.available is True
        assert status.detail == DEFAULT_MODEL
        assert sdk.calls[0]["max_tokens"] == 1

    async def test_a_failed_request_is_not_ready(self, sdk: FakeSDK) -> None:
        sdk.error = RuntimeError("Error code: 401 - authentication_error")

        status = await DeepSeekProvider(api_key=DS_KEY).health_check()

        assert status.available is False


class TestErrors:
    async def test_missing_key_names_the_deepseek_secret(self, sdk: FakeSDK) -> None:
        with pytest.raises(ProviderError) as caught:
            await DeepSeekProvider().complete("s", "u", DetectionConfig())

        assert "No DeepSeek API key is set." in str(caught.value)
        assert "autoclip config set-secret deepseek" in caught.value.hint
        assert sdk.clients == []

    @pytest.mark.parametrize(
        ("raw", "message", "hint"),
        [
            (
                "Error code: 401 - authentication_error",
                "DeepSeek rejected the API key.",
                "autoclip config set-secret deepseek",
            ),
            ("Error code: 429 - rate limit exceeded", "DeepSeek rate limit reached.", "retry"),
            (
                "Error code: 404 - not_found_error: model",
                f"DeepSeek does not recognise the model '{DEFAULT_MODEL}'.",
                DEFAULT_MODEL,
            ),
            (
                "Error code: 402 - {'error': {'message': 'Insufficient Balance'}}",
                "DeepSeek reports a billing or credit problem.",
                "platform.deepseek.com",
            ),
        ],
    )
    async def test_errors_are_worded_for_deepseek(
        self, sdk: FakeSDK, raw: str, message: str, hint: str
    ) -> None:
        sdk.error = RuntimeError(raw)

        with pytest.raises(ProviderError) as caught:
            await DeepSeekProvider(api_key=DS_KEY).complete("s", "u", DetectionConfig())

        assert str(caught.value).startswith(message)
        assert hint in caught.value.hint
        assert caught.value.provider == "deepseek"

    async def test_anthropic_wording_is_unchanged(self, sdk: FakeSDK) -> None:
        sdk.error = RuntimeError("Error code: 401 - authentication_error")

        with pytest.raises(ProviderError) as caught:
            await AnthropicProvider(api_key=ANT_KEY).complete("s", "u", DetectionConfig())

        assert str(caught.value).startswith("Anthropic rejected the API key.")
        assert "autoclip config set-secret anthropic" in caught.value.hint


class TestSecret:
    def test_set_secret_cli_accepts_deepseek(self, fake_keyring) -> None:
        from autoclip.cli import app
        from typer.testing import CliRunner

        result = CliRunner().invoke(app, ["config", "set-secret", "deepseek"], input=f"{DS_KEY}\n")

        assert result.exit_code == 0, result.output
        assert fake_keyring.store[(config.KEYRING_SERVICE, "deepseek")] == DS_KEY
        assert DS_KEY not in result.output

    def test_api_stores_it_write_only(self, client: TestClient, fake_keyring) -> None:
        response = client.put("/api/settings/secrets", json={"key": "deepseek", "value": DS_KEY})

        assert response.status_code == 204
        body = client.get("/api/settings").json()
        assert body["keys_present"]["deepseek"] is True
        assert DS_KEY not in str(body)

    def test_env_override_is_honoured(self, fake_keyring, monkeypatch) -> None:
        monkeypatch.setenv("AUTOCLIP_DEEPSEEK_KEY", DS_KEY)

        assert build_provider("deepseek", config.Settings()).api_key == DS_KEY

    async def test_the_key_is_never_logged(
        self,
        sdk: FakeSDK,
        fake_keyring,
        window: TranscriptWindow,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.DEBUG)
        config.set_secret("deepseek", DS_KEY)
        provider = build_provider("deepseek", config.load())
        sdk.error = RuntimeError("Error code: 401 - authentication_error")

        with pytest.raises(ProviderError) as caught:
            await provider.detect_highlights(window, DetectionConfig())
        status = await provider.health_check()

        assert DS_KEY not in caplog.text
        assert DS_KEY not in str(caught.value)
        assert DS_KEY not in status.detail

    def test_anthropic_env_credentials_never_reach_deepseek(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The real SDK client, built but never used: the environment's Anthropic
        # endpoint and credentials must not leak into DeepSeek requests.
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic-env.invalid")
        monkeypatch.setenv("ANTHROPIC_API_KEY", ANT_KEY)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", ANT_KEY)

        sdk_client = DeepSeekProvider(api_key=DS_KEY)._client()

        assert str(sdk_client.base_url).rstrip("/") == DEFAULT_BASE_URL
        assert sdk_client.auth_headers == {"X-Api-Key": DS_KEY}


class TestSelectable:
    def test_can_be_the_active_provider(self, client: TestClient) -> None:
        body = client.put("/api/settings", json={"active_provider": "deepseek"}).json()

        assert body["active_provider"] == "deepseek"
        assert config.load().active_provider == "deepseek"

    def test_per_job_choice_reaches_the_runner(self, client: TestClient) -> None:
        source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4"))

        response = client.post(
            "/api/jobs", json={"source_id": source.id, "settings": {"provider": "deepseek"}}
        )

        assert response.status_code == 201
        job = store.get_job(response.json()["id"])
        assert job.provider == "deepseek"
        assert settings_for_job(job).active_provider == "deepseek"
        assert config.load().active_provider == "anthropic"  # saved settings untouched

    def test_status_shows_ready_or_not(self, client: TestClient, fake_keyring, sdk) -> None:
        def deepseek_status() -> dict[str, Any]:
            statuses = client.get("/api/providers/status").json()
            return next(s for s in statuses if s["name"] == "deepseek")

        before = deepseek_status()
        client.put("/api/settings/secrets", json={"key": "deepseek", "value": DS_KEY})
        after = deepseek_status()

        assert before["requires_key"] is True
        assert (before["has_key"], before["available"]) == (False, False)
        assert (after["has_key"], after["available"]) == (True, True)
        assert after["detail"] == DEFAULT_MODEL
