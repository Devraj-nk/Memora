from types import SimpleNamespace

import pytest

import memora.llm.client as client_module
from memora.config import settings


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.response = text


class _FakeOllamaClient:
    def __init__(self, host: str | None = None) -> None:
        self.host = host
        self.last_call: tuple[str, str] | None = None

    def generate(self, model: str, prompt: str) -> _FakeResponse:
        self.last_call = (model, prompt)
        return _FakeResponse(f"answer to: {prompt}")


@pytest.fixture(autouse=True)
def _reset_client_cache():
    client_module._get_client.cache_clear()
    yield
    client_module._get_client.cache_clear()


def test_generate_returns_the_client_response_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "ollama", SimpleNamespace(Client=_FakeOllamaClient))

    result = client_module.generate("What is Memora?")

    assert result == "answer to: What is Memora?"


def test_generate_uses_configured_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOllamaClient()
    monkeypatch.setattr(client_module, "ollama", SimpleNamespace(Client=lambda host=None: fake))

    client_module.generate("hello")

    assert fake.last_call is not None
    assert fake.last_call[0] == settings.llm_model


def test_generate_accepts_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOllamaClient()
    monkeypatch.setattr(client_module, "ollama", SimpleNamespace(Client=lambda host=None: fake))

    client_module.generate("hello", model_name="custom-model")

    assert fake.last_call is not None
    assert fake.last_call[0] == "custom-model"
