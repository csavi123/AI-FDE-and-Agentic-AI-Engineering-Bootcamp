"""Tests for GET /debug/retrieve. Pinecone and Claude are faked, so these cost nothing."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pinecone.exceptions import PineconeApiException

import app

client = TestClient(app.app)
AUTH = {"X-App-Token": "test-token"}


@pytest.fixture(autouse=True)
def reset_rate_limit():
    app.limiter.reset()


def stored(*matches):
    return SimpleNamespace(matches=[
        SimpleNamespace(id=f"{doc}#{i}", score=score,
                        metadata={"text": text, "document_id": doc, "chunk_index": float(i), "source": f"{doc}.txt"})
        for doc, i, score, text in matches
    ])


def test_returns_ranked_chunks_with_scores_and_document_ids(fake_pinecone):
    fake_pinecone.index.query.return_value = stored(
        ("POL-101", 1, 0.62, "Remote work: up to 3 days per week."),
        ("POL-114", 0, 0.18, "Mileage is 45p per mile."),
    )

    with patch.object(app.client.messages, "create") as claude:
        r = client.get("/debug/retrieve", params={"q": "How many remote days?"}, headers=AUTH)

    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "How many remote days?"
    assert [(m["document_id"], m["chunk_index"], m["score"]) for m in body["matches"]] == [
        ("POL-101", 1, 0.62), ("POL-114", 0, 0.18),
    ]
    assert body["matches"][0]["text"] == "Remote work: up to 3 days per week."
    assert body["matches"][0]["source"] == "POL-101.txt"
    assert body["tokens_used"] == 3
    claude.assert_not_called()  # retrieval only, never the LLM


def test_embeds_question_as_query_and_asks_for_top_5_by_default(fake_pinecone):
    fake_pinecone.index.query.return_value = stored()

    r = client.get("/debug/retrieve", params={"q": "  remote days  "}, headers=AUTH)

    assert r.json()["top_k"] == 5
    assert fake_pinecone.index.query.call_args.kwargs["top_k"] == 5
    embed_kwargs = fake_pinecone.embed.call_args.kwargs
    assert embed_kwargs["inputs"] == ["remote days"]
    assert embed_kwargs["parameters"]["input_type"] == "query"


def test_top_k_can_be_changed(fake_pinecone):
    fake_pinecone.index.query.return_value = stored()

    client.get("/debug/retrieve", params={"q": "remote", "top_k": 2}, headers=AUTH)

    assert fake_pinecone.index.query.call_args.kwargs["top_k"] == 2


@pytest.mark.parametrize("params, status", [
    ({"q": ""}, 400),
    ({"q": "   "}, 400),
    ({}, 422),                         # q missing
    ({"q": "x", "top_k": 0}, 422),
    ({"q": "x", "top_k": 21}, 422),
])
def test_bad_query_params(fake_pinecone, params, status):
    r = client.get("/debug/retrieve", params=params, headers=AUTH)

    assert r.status_code == status
    fake_pinecone.embed.assert_not_called()


def test_needs_token(fake_pinecone):
    assert client.get("/debug/retrieve", params={"q": "remote"}).status_code == 401
    fake_pinecone.embed.assert_not_called()


def test_pinecone_error_is_502(fake_pinecone):
    fake_pinecone.index.query.side_effect = PineconeApiException(status=500, reason="Internal Server Error")

    r = client.get("/debug/retrieve", params={"q": "remote"}, headers=AUTH)

    assert r.status_code == 502
