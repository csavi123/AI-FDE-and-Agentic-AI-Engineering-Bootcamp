"""Tests for retrieval-augmented /ask and /ask/stream. Pinecone (see conftest.py) and Claude are faked."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pinecone.exceptions import PineconeApiException

import app

client = TestClient(app.app)
AUTH = {"X-App-Token": "test-token"}
QUESTION = "How many remote days are allowed?"


@pytest.fixture(autouse=True)
def reset_rate_limit():
    app.limiter.reset()


def stored(*matches):
    """Fake Pinecone query result: (document_id, chunk_index, score, text) per match, best first."""
    return SimpleNamespace(matches=[
        SimpleNamespace(id=f"{doc}#{i}", score=score,
                        metadata={"text": text, "document_id": doc, "chunk_index": float(i), "source": f"{doc}.txt"})
        for doc, i, score, text in matches
    ])


HANDBOOK = stored(
    ("POL-101", 5, 0.62, "Remote work. Employees may work remotely up to three days per week."),
    ("POL-101", 6, 0.58, "Home office standards. Remote employees need a private space."),
    ("POL-118", 1, 0.21, "Book rooms in the Workplace app."),
)


def claude_reply(answer, citations, input_tokens=500, output_tokens=40):
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=json.dumps({"answer": answer, "citations": citations}))],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def ask(question=QUESTION):
    return client.post("/ask", json={"question": question}, headers=AUTH)


# --- Retrieval feeds generation ----------------------------------------------

def test_answer_is_grounded_in_retrieved_chunks(fake_pinecone):
    fake_pinecone.index.query.return_value = HANDBOOK
    reply = claude_reply("Up to three days per week [POL-101].", ["POL-101"])
    with patch.object(app.client.messages, "create", return_value=reply) as claude:
        r = ask()

    assert r.status_code == 200
    assert r.json() == {
        "answer": "Up to three days per week [POL-101].",
        "citations": ["POL-101"],
        "retrieved_chunk_ids": ["POL-101#5", "POL-101#6", "POL-118#1"],
        "tokens_used": 543,                # 500 + 40 Claude, plus 3 to embed the question
        "cost_usd": 0.00007048,            # (500 * 0.10 + 40 * 0.50) / 1M + 3 * 0.16 / 1M
    }
    assert fake_pinecone.index.query.call_args.kwargs["top_k"] == 5
    assert fake_pinecone.embed.call_args.kwargs["parameters"]["input_type"] == "query"

    prompt = claude.call_args.kwargs["messages"][0]["content"]
    assert prompt.startswith("Answer using ONLY the context below.")
    assert app.REFUSAL_ANSWER in prompt
    assert "[chunk POL-101#5] document_id: POL-101 (source: POL-101.txt)\n" \
           "Remote work. Employees may work remotely up to three days per week." in prompt
    assert "[chunk POL-118#1] document_id: POL-118" in prompt
    assert prompt.endswith("Question: How many remote days are allowed?")
    assert claude.call_args.kwargs["system"] == app.SYSTEM_PROMPT


def test_grounding_prompt_matches_template():
    chunk = {"id": "POL-114#2", "document_id": "POL-114", "source": "doc2_expenses.txt", "text": "Mileage is 45p."}

    assert app.grounding_prompt("What is the mileage rate?", [chunk]) == (
        "Answer using ONLY the context below.\n"
        "If the context does not contain the answer, say:\n"
        "\"I don't have enough information to answer that.\"\n"
        "Cite the document_id of each chunk you used.\n"
        "\n"
        "Context:\n"
        "[chunk POL-114#2] document_id: POL-114 (source: doc2_expenses.txt)\n"
        "Mileage is 45p.\n"
        "\n"
        "Question: What is the mileage rate?"
    )


def test_refusal_from_model_passes_through(fake_pinecone):
    fake_pinecone.index.query.return_value = HANDBOOK
    with patch.object(app.client.messages, "create", return_value=claude_reply(app.REFUSAL_ANSWER, [])):
        r = ask("What is the parental leave policy?")

    assert r.status_code == 200
    assert r.json()["answer"] == "I don't have enough information to answer that."
    assert r.json()["citations"] == []
    assert r.json()["retrieved_chunk_ids"] == ["POL-101#5", "POL-101#6", "POL-118#1"]


def test_empty_index_refuses_without_calling_claude(fake_pinecone):
    fake_pinecone.index.query.return_value = stored()
    with patch.object(app.client.messages, "create") as claude:
        r = ask()

    assert r.json() == {
        "answer": app.REFUSAL_ANSWER, "citations": [], "retrieved_chunk_ids": [],
        "tokens_used": 3, "cost_usd": 0.00000048,
    }
    claude.assert_not_called()


# --- Citation guardrail --------------------------------------------------------

def test_citing_a_document_that_was_not_retrieved_is_502(fake_pinecone):
    fake_pinecone.index.query.return_value = HANDBOOK
    with patch.object(app.client.messages, "create", return_value=claude_reply("Three days.", ["POL-999"])):
        r = ask()

    assert r.status_code == 502
    assert r.json()["detail"] == "Model cited a document that was not retrieved."


def test_chunk_id_citations_become_document_ids_once(fake_pinecone):
    fake_pinecone.index.query.return_value = HANDBOOK
    reply = claude_reply("Three days.", ["POL-101#5", "POL-101", "POL-101#6"])
    with patch.object(app.client.messages, "create", return_value=reply):
        r = ask()

    assert r.json()["citations"] == ["POL-101"]


# --- Order of checks and retrieval errors ------------------------------------------

def test_token_is_checked_before_retrieval(fake_pinecone):
    with patch.object(app.client.messages, "create") as claude:
        r = client.post("/ask", json={"question": QUESTION})

    assert r.status_code == 401
    fake_pinecone.embed.assert_not_called()
    claude.assert_not_called()


def test_retrieval_error_stops_before_claude(fake_pinecone):
    fake_pinecone.index.query.side_effect = PineconeApiException(status=500, reason="Internal Server Error")
    with patch.object(app.client.messages, "create") as claude:
        r = ask()

    assert r.status_code == 502
    assert r.json()["detail"] == "Vector store request failed."
    claude.assert_not_called()


def test_missing_pinecone_config_is_503(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY")
    app.rag.pinecone_client.cache_clear()
    app.rag.pinecone_index.cache_clear()
    with patch.object(app.client.messages, "create") as claude:
        r = ask()

    assert r.status_code == 503
    assert r.json()["detail"] == "PINECONE_API_KEY is not set."
    claude.assert_not_called()


# --- Streaming --------------------------------------------------------------

class FakeStreamManager:
    def __init__(self, final):
        self.final = final

    def __enter__(self):
        text = self.final.content[0].text
        return SimpleNamespace(text_stream=iter([text[:10], text[10:]]), get_final_message=lambda: self.final)

    def __exit__(self, *exc):
        pass


def sse_events(response):
    return [
        (dict(line.split(": ", 1) for line in block.splitlines())["event"],
         json.loads(dict(line.split(": ", 1) for line in block.splitlines())["data"]))
        for block in response.text.strip().split("\n\n")
    ]


def test_stream_uses_the_same_grounded_path(fake_pinecone):
    fake_pinecone.index.query.return_value = HANDBOOK
    reply = claude_reply("Up to three days per week [POL-101].", ["POL-101"])
    with patch.object(app.client.messages, "stream", return_value=FakeStreamManager(reply)) as claude:
        r = client.post("/ask/stream", json={"question": QUESTION}, headers=AUTH)

    events = sse_events(r)
    assert "".join(d["text"] for e, d in events if e == "delta") == "Up to three days per week [POL-101]."
    event, done = events[-1]
    assert event == "done"
    assert done["citations"] == ["POL-101"]
    assert done["retrieved_chunk_ids"] == ["POL-101#5", "POL-101#6", "POL-118#1"]
    assert claude.call_args.kwargs["messages"][0]["content"].startswith("Answer using ONLY the context below.")


def test_stream_with_empty_index_sends_refusal_without_calling_claude(fake_pinecone):
    fake_pinecone.index.query.return_value = stored()
    with patch.object(app.client.messages, "stream") as claude:
        r = client.post("/ask/stream", json={"question": QUESTION}, headers=AUTH)

    assert sse_events(r) == [("done", {
        "answer": app.REFUSAL_ANSWER, "citations": [], "retrieved_chunk_ids": [],
        "tokens_used": 3, "cost_usd": 0.00000048,
    })]
    claude.assert_not_called()


def test_stream_retrieval_error_is_a_plain_http_error(fake_pinecone):
    fake_pinecone.index.query.side_effect = PineconeApiException(status=500, reason="Internal Server Error")
    with patch.object(app.client.messages, "stream") as claude:
        r = client.post("/ask/stream", json={"question": QUESTION}, headers=AUTH)

    assert r.status_code == 502
    claude.assert_not_called()
