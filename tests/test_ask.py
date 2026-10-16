"""Tests for POST /ask. Claude is replaced with a fake, so these never call Anthropic or cost money."""
import json
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


# --- Streaming: POST /ask/stream ---------------------------------------------

class FakeStream:
    """Stand-in for the SDK's MessageStream: yields text chunks, then the final message."""
    def __init__(self, chunks, final=None, error=None):
        self._chunks, self._final, self._error = chunks, final, error

    @property
    def text_stream(self):
        yield from self._chunks
        if self._error:
            raise self._error

    def get_final_message(self):
        return self._final


class FakeStreamManager:
    """Stand-in for client.messages.stream(...); opening it can raise like the real one."""
    def __init__(self, stream=None, open_error=None):
        self.stream, self.open_error, self.closed = stream, open_error, False

    def __enter__(self):
        if self.open_error:
            raise self.open_error
        return self.stream

    def __exit__(self, *exc):
        self.closed = True


def stream_of(text, chunk_size=7, **final_kwargs):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return FakeStreamManager(FakeStream(chunks, final=fake_message(text=text, **final_kwargs)))


def sse_events(response):
    """Parse a server-sent-events body into [(event, data), ...]."""
    events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def post_stream(body=None, headers=AUTH):
    return client.post("/ask/stream", json=body or {"question": "What is RAG?"}, headers=headers)


def test_stream_sends_deltas_then_validated_done():
    reply = '{"answer": "RAG is \\"retrieval\\" plus generation."}'
    manager = stream_of(reply, chunk_size=5)
    with patch.object(app.client.messages, "stream", return_value=manager):
        r = post_stream()

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r)
    deltas = "".join(d["text"] for e, d in events if e == "delta")
    assert deltas == 'RAG is "retrieval" plus generation.'
    assert events[-1] == ("done", {
        "answer": 'RAG is "retrieval" plus generation.',
        "tokens_used": 120,
        "cost_usd": 0.000052,
    })
    assert manager.closed


def test_stream_malformed_output_ends_with_error_event():
    manager = stream_of('{"answer": "RAG is retrieval')  # never closed
    with patch.object(app.client.messages, "stream", return_value=manager):
        events = sse_events(post_stream())

    assert events[-1] == ("error", {"status": 502, "detail": "Model returned malformed output."})
    assert not any(e == "done" for e, _ in events)


def test_stream_wrong_key_streams_nothing_and_errors():
    with patch.object(app.client.messages, "stream", return_value=stream_of('{"respuesta": "hola"}')):
        events = sse_events(post_stream())

    assert events == [("error", {"status": 502, "detail": "Model returned malformed output."})]


def test_stream_refusal_ends_with_422_event():
    manager = stream_of("", stop_reason="refusal")
    manager.stream._chunks = []
    with patch.object(app.client.messages, "stream", return_value=manager):
        events = sse_events(post_stream())

    assert events == [("error", {"status": 422, "detail": "The model declined this question."})]


def test_stream_interrupted_midway_ends_with_error_event():
    broken = FakeStream(['{"answer": "RAG'], error=anthropic.APIConnectionError(request=ANTHROPIC_REQUEST))
    manager = FakeStreamManager(broken)
    with patch.object(app.client.messages, "stream", return_value=manager):
        events = sse_events(post_stream())

    assert events[-1] == ("error", {"status": 502, "detail": "The model stream was interrupted."})
    assert manager.closed


@pytest.mark.parametrize("error, expected_status", [
    (api_error(anthropic.AuthenticationError, 401), 500),
    (api_error(anthropic.RateLimitError, 429), 429),
    (anthropic.APIConnectionError(request=ANTHROPIC_REQUEST), 502),
])
def test_stream_errors_when_opening_are_plain_http_errors(error, expected_status):
    with patch.object(app.client.messages, "stream", return_value=FakeStreamManager(open_error=error)):
        r = post_stream()

    assert r.status_code == expected_status


def test_stream_checks_token_and_input_before_calling_claude():
    with patch.object(app.client.messages, "stream") as fake:
        assert post_stream(headers={"X-App-Token": "wrong"}).status_code == 401
        assert post_stream({"question": "   "}).status_code == 422
        assert post_stream({"question": "hi", "force_bad": True}).status_code == 403
    fake.assert_not_called()


@pytest.mark.parametrize("raw, expected", [
    ('', ''),
    ('{"answ', ''),
    ('{"answer": "', ''),
    ('{"answer": "RAG is', 'RAG is'),
    ('{"answer": "a \\', 'a '),                          # escape not finished yet
    ('{"answer": "a \\"b\\" c', 'a "b" c'),
    ('{"answer": "caf\\u00', 'caf'),                     # \u escape not finished yet
    ('{"answer": "caf\\u00e9", "x', 'café'),
    ('{"answer": "hi \\ud83d', 'hi '),                   # emoji: waiting for the low surrogate
    ('{"answer": "hi \\ud83d\\ude00"}', 'hi 😀'),
    ('```json\n{"answer": "fenced', 'fenced'),
    ('{"respuesta": "hola"}', ''),
])
def test_partial_answer(raw, expected):
    assert app.partial_answer(raw) == expected
