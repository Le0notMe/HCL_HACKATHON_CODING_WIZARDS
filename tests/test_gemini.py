"""Tests for the Gemini adapter, through the real SDK against a local stub server."""

import json

import pytest

from app.config import Settings
from app.llm import GeminiLLM, LLMError, build_llm
from tests.test_llm import CLASSIFICATION, TICKET, StubServer


def reply(text, finish_reason="STOP"):
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish_reason}
        ]
    }


def error(code, status, message="failed"):
    return {"error": {"code": code, "message": message, "status": status}}


@pytest.fixture
def stub():
    server = StubServer()
    server.body = reply(json.dumps(CLASSIFICATION))
    yield server
    server.close()


def settings_for(url):
    return Settings(_env_file=None, gemini_api_key="test-key", gemini_base_url=url, llm_max_retries=0)


@pytest.fixture
def llm(stub):
    return GeminiLLM(settings_for(stub.url))


def test_gemini_is_the_default_provider(stub):
    assert isinstance(build_llm(settings_for(stub.url)), GeminiLLM)


def test_key_from_settings_is_sent(llm, stub):
    llm.classify(TICKET)
    headers = {key.lower(): value for key, value in stub.requests[0]["headers"].items()}
    assert headers["x-goog-api-key"] == "test-key"


def test_classify_sends_a_structured_request_and_parses_the_reply(llm, stub):
    assert llm.classify(TICKET).subcategory == "VPN Authentication"

    request = stub.requests[0]
    body = request["body"]
    config = body["generationConfig"]
    assert "gemini-flash-latest:generateContent" in request["path"]
    assert config["responseMimeType"] == "application/json"
    assert "confidence" in json.dumps(config["responseSchema"])
    assert "taxonomy" in body["systemInstruction"]["parts"][0]["text"]
    assert "tools" not in body


def test_ticket_text_is_sent_as_data_that_cannot_close_its_block(llm, stub):
    llm.classify(TICKET)
    prompt = stub.requests[0]["body"]["contents"][0]["parts"][0]["text"]
    assert prompt.count("</untrusted_data>") == 1


def test_synthesize_parses_nested_output(llm, stub):
    stub.body = reply(
        json.dumps(
            {
                "resolution_paths": [
                    {
                        "summary": "Re-issue",
                        "steps": ["Re-issue the profile"],
                        "evidence": [{"source_id": "INC-1", "excerpt": "Re-issued it", "reason": "same error"}],
                    }
                ],
                "evidence_conflict": False,
                "clarifying_questions": [],
            }
        )
    )
    output = llm.synthesize(TICKET, [{"source_id": "INC-1", "text": "Re-issued it", "resolution": "Re-issued it"}])
    assert output.resolution_paths[0].evidence[0].source_id == "INC-1"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (200, {"promptFeedback": {"blockReason": "SAFETY"}}, "declined"),
        (200, reply("", "SAFETY"), "declined"),
        (200, reply('{"category":', "MAX_TOKENS"), "truncated"),
        (200, reply('{"category": "only this field"}'), "invalid"),
        (200, reply("not json"), "invalid"),
        (200, {"candidates": []}, "no structured output"),
        (400, error(400, "INVALID_ARGUMENT", "API key not valid. API_KEY_INVALID"), "credentials"),
        (403, error(403, "PERMISSION_DENIED"), "credentials"),
        (400, error(400, "INVALID_ARGUMENT"), "rejected"),
        (404, error(404, "NOT_FOUND"), "rejected"),
        (429, error(429, "RESOURCE_EXHAUSTED"), "rate limited"),
        (500, error(500, "INTERNAL"), "status 500"),
        (503, error(503, "UNAVAILABLE"), "status 503"),
    ],
)
def test_every_failure_becomes_llm_error(llm, stub, status, body, expected):
    stub.status, stub.body = status, body
    with pytest.raises(LLMError, match=expected):
        llm.classify(TICKET)


@pytest.fixture
def fallback_llm(stub):
    settings = settings_for(stub.url).model_copy(
        update={"llm_model": "primary", "llm_fallback_models": ["backup", "last"]}
    )
    return GeminiLLM(settings)


def models_called(stub):
    return [request["path"].rsplit("/", 1)[1].split(":")[0] for request in stub.requests]


@pytest.mark.parametrize(
    ("status", "name"), [(503, "UNAVAILABLE"), (429, "RESOURCE_EXHAUSTED")]
)
def test_busy_model_falls_back_to_the_next(fallback_llm, stub, status, name):
    stub.failing_models = {"primary": (status, error(status, name))}
    assert fallback_llm.classify(TICKET).subcategory == "VPN Authentication"
    assert models_called(stub) == ["primary", "backup"]


def test_healthy_model_uses_no_fallback(fallback_llm, stub):
    fallback_llm.classify(TICKET)
    assert models_called(stub) == ["primary"]


def test_all_models_busy_becomes_llm_error(fallback_llm, stub):
    stub.status, stub.body = 503, error(503, "UNAVAILABLE")
    with pytest.raises(LLMError, match="status 503"):
        fallback_llm.classify(TICKET)
    assert models_called(stub) == ["primary", "backup", "last"]


def test_rejected_credentials_do_not_fall_back(fallback_llm, stub):
    stub.status, stub.body = 403, error(403, "PERMISSION_DENIED")
    with pytest.raises(LLMError, match="credentials"):
        fallback_llm.classify(TICKET)
    assert models_called(stub) == ["primary"]


def test_unreachable_server_becomes_llm_error():
    llm = GeminiLLM(settings_for("http://127.0.0.1:9"))
    with pytest.raises(LLMError, match="connection failed"):
        llm.classify(TICKET)
