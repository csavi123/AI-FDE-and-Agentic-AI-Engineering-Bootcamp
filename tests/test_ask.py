"""Tests for POST /ask. Claude is replaced with a fake, so these never call Anthropic or cost money."""
from types import SimpleNamespace
from unittest.mock import patch

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

import app

client = TestClient(app.app)
AUTH = {"X-App-Token": "test-token"}
ANTHROPIC_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def fake_message(text='{"answer": "RAG retrieves documents and feeds them to the model."}',
                 input_tokens=20, output_tokens=100, stop_reason="end_turn"):
    """A stand-in for the object client.messages.create() returns."""
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def api_error(cls, status):
    return cls("error", response=httpx.Response(status, request=ANTHROPIC_REQUEST), body=None)


@pytest.fixture(autouse=True)
def reset_rate_limit():
    app.limiter.reset()


@pytest.fixture
def fake_claude():
    with patch.object(app.client.messages, "create", return_value=fake_message()) as mock:
        yield mock


# --- Happy path -------------------------------------------------------------

def test_returns_answer_tokens_and_cost(fake_claude):
    r = client.post("/ask", json={"question": "What is RAG in one sentence?"}, headers=AUTH)

    assert r.status_code == 200
    assert r.json() == {
        "answer": "RAG retrieves documents and feeds them to the model.",
        "tokens_used": 120,
        "cost_usd": 0.000052,  # (20 * 0.10 + 100 * 0.50) / 1M
    }
    sent = fake_claude.call_args.kwargs
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["messages"] == [{"role": "user", "content": "What is RAG in one sentence?"}]
    assert sent["system"] == app.SYSTEM_PROMPT


def test_question_is_trimmed_before_sending(fake_claude):
    client.post("/ask", json={"question": "  What is RAG?  "}, headers=AUTH)

    assert fake_claude.call_args.kwargs["messages"][0]["content"] == "What is RAG?"


# --- Validation: rejected with 422 before Claude is called -------------------

@pytest.mark.parametrize("body", [
    {},                          # missing question
    {"question": ""},            # empty
    {"question": "   "},         # only whitespace
    {"question": "a" * 4001},    # too long
    {"question": 123},           # wrong type
])
def test_invalid_question_is_rejected_without_calling_claude(fake_claude, body):
    r = client.post("/ask", json=body, headers=AUTH)

    assert r.status_code == 422
    fake_claude.assert_not_called()


def test_max_length_question_is_accepted(fake_claude):
    r = client.post("/ask", json={"question": "a" * 4000}, headers=AUTH)

    assert r.status_code == 200


# --- Auth --------------------------------------------------------------------

@pytest.mark.parametrize("headers", [{}, {"X-App-Token": "wrong"}])
def test_missing_or_wrong_token_is_rejected(fake_claude, headers):
    r = client.post("/ask", json={"question": "hi"}, headers=headers)

    assert r.status_code == 401
    fake_claude.assert_not_called()


# --- Rate limiting -----------------------------------------------------------

def test_eleventh_request_in_a_minute_is_rate_limited(fake_claude):
    codes = [client.post("/ask", json={"question": "hi"}, headers=AUTH).status_code for _ in range(11)]

    assert codes == [200] * 10 + [429]
    assert fake_claude.call_count == 10


def test_rate_limit_also_throttles_token_guessing():
    codes = [client.post("/ask", json={"question": "hi"}, headers={"X-App-Token": "guess"}).status_code
             for _ in range(11)]

    assert codes == [401] * 10 + [429]


# --- Model refusals and upstream errors --------------------------------------

def test_refusal_returns_422():
    with patch.object(app.client.messages, "create", return_value=fake_message(text="", stop_reason="refusal")):
        r = client.post("/ask", json={"question": "hi"}, headers=AUTH)

    assert r.status_code == 422
    assert r.json()["detail"] == "The model declined this question."


@pytest.mark.parametrize("error, expected_status", [
    (api_error(anthropic.AuthenticationError, 401), 500),
    (api_error(anthropic.RateLimitError, 429), 429),
    (api_error(anthropic.InternalServerError, 500), 502),
    (anthropic.APIConnectionError(request=ANTHROPIC_REQUEST), 502),
])
def test_upstream_errors_are_mapped(error, expected_status):
    with patch.object(app.client.messages, "create", side_effect=error):
        r = client.post("/ask", json={"question": "hi"}, headers=AUTH)

    assert r.status_code == expected_status


# --- Output guardrail: the model's reply must match the JSON contract -------

@pytest.mark.parametrize("model_text", [
    '{"answer": "RAG is retrieval plus generation."',     # missing closing brace
    '{"respuesta": "RAG is retrieval plus generation."}',  # wrong key
    '{"answer": ""}',                                      # empty answer
    '{"answer": "ok", "extra": 1}',                        # unexpected field
    "RAG is retrieval plus generation.",                   # not JSON at all
])
def test_malformed_model_output_returns_502(model_text):
    with patch.object(app.client.messages, "create", return_value=fake_message(text=model_text)):
        r = client.post("/ask", json={"question": "hi"}, headers=AUTH)

    assert r.status_code == 502
    assert r.json()["detail"] == "Model returned malformed output."


def test_code_fenced_json_is_accepted():
    fenced = '```json\n{"answer": "RAG is retrieval plus generation."}\n```'
    with patch.object(app.client.messages, "create", return_value=fake_message(text=fenced)):
        r = client.post("/ask", json={"question": "hi"}, headers=AUTH)

    assert r.status_code == 200
    assert r.json()["answer"] == "RAG is retrieval plus generation."


# --- force_bad switch --------------------------------------------------------

def test_force_bad_is_refused_when_disabled(fake_claude):
    r = client.post("/ask", json={"question": "hi", "force_bad": True}, headers=AUTH)

    assert r.status_code == 403
    fake_claude.assert_not_called()


def test_force_bad_adds_instruction_and_bad_output_is_caught(monkeypatch):
    monkeypatch.setattr(app, "ALLOW_FORCE_BAD", True)
    broken = fake_message(text='{"answer": "RAG is')
    with patch.object(app.client.messages, "create", return_value=broken) as fake:
        r = client.post("/ask", json={"question": "hi", "force_bad": True}, headers=AUTH)

    assert app.FORCE_BAD_INSTRUCTION in fake.call_args.kwargs["system"]
    assert r.status_code == 502


def test_normal_request_has_no_force_bad_instruction(fake_claude):
    client.post("/ask", json={"question": "hi"}, headers=AUTH)

    assert app.FORCE_BAD_INSTRUCTION not in fake_claude.call_args.kwargs["system"]


# --- Health check ------------------------------------------------------------

def test_health_check_needs_no_token():
    r = client.get("/")

    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
