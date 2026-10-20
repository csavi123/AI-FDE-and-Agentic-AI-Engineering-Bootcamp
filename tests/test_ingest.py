"""Tests for POST /ingest. Pinecone is faked (see conftest.py), so these cost nothing."""
import pytest
import urllib3
from fastapi.testclient import TestClient
from pinecone.exceptions import PineconeApiException, UnauthorizedException

import app
import rag

client = TestClient(app.app)
AUTH = {"X-App-Token": "test-token"}
LONG_TEXT = "\n\n".join(f"Paragraph {n}. " + "Refunds are available within thirty days. " * 8 for n in range(10))


@pytest.fixture(autouse=True)
def reset_rate_limit():
    app.limiter.reset()


def upserted(fake_pinecone):
    return [v for c in fake_pinecone.index.upsert.call_args_list for v in c.kwargs["vectors"]]


# --- Happy path -------------------------------------------------------------

def test_ingest_short_document(fake_pinecone):
    r = client.post("/ingest", headers=AUTH, json={
        "document_id": "refund-policy", "text": "Refunds within 30 days.", "source": "refund_policy.md",
    })

    assert r.status_code == 200
    assert r.json() == {
        "document_id": "refund-policy",
        "chunks_indexed": 1,
        "status": "indexed",
        "tokens_used": 3,
        "cost_usd": 0.00000048,  # 3 tokens * $0.16 / 1M
    }
    [vector] = upserted(fake_pinecone)
    assert vector["id"] == "refund-policy#0"
    assert vector["metadata"] == {
        "document_id": "refund-policy", "chunk_index": 0, "source": "refund_policy.md",
        "text": "Refunds within 30 days.",
    }
    assert fake_pinecone.embed.call_args.kwargs["model"] == "llama-text-embed-v2"
    assert fake_pinecone.embed.call_args.kwargs["parameters"]["input_type"] == "passage"


def test_long_document_is_chunked_with_ordered_ids_and_metadata(fake_pinecone):
    r = client.post("/ingest", headers=AUTH, json={
        "document_id": "policy", "text": LONG_TEXT, "metadata": {"department": "support"},
    })

    vectors = upserted(fake_pinecone)
    assert r.json()["chunks_indexed"] == len(vectors) > 1
    for i, v in enumerate(vectors):
        assert v["id"] == f"policy#{i}"
        assert v["metadata"]["chunk_index"] == i
        assert v["metadata"]["document_id"] == "policy"
        assert v["metadata"]["source"] == "policy"  # defaults to document_id
        assert v["metadata"]["department"] == "support"
        assert len(v["metadata"]["text"]) <= rag.CHUNK_SIZE


def test_chunk_size_and_overlap_are_configurable(monkeypatch):
    monkeypatch.setattr(rag, "CHUNK_SIZE", 50)
    monkeypatch.setattr(rag, "CHUNK_OVERLAP", 10)

    chunks = rag.split_text("word " * 100)

    assert all(len(c) <= 50 for c in chunks)
    assert len(chunks) > 10


def test_reingesting_a_shorter_version_deletes_leftover_chunks(fake_pinecone):
    fake_pinecone.index.list.return_value = iter([["doc#0", "doc#1", "doc#2"]])

    client.post("/ingest", headers=AUTH, json={"document_id": "doc", "text": "Now just one chunk."})

    assert fake_pinecone.index.list.call_args.kwargs == {"prefix": "doc#"}
    fake_pinecone.index.delete.assert_called_once_with(ids=["doc#1", "doc#2"])


def test_reingesting_same_length_deletes_nothing(fake_pinecone):
    fake_pinecone.index.list.return_value = iter([["doc#0"]])

    client.post("/ingest", headers=AUTH, json={"document_id": "doc", "text": "Still one chunk."})

    fake_pinecone.index.delete.assert_not_called()


# --- Bad input ----------------------------------------------------------------

@pytest.mark.parametrize("body, detail", [
    ({"document_id": "doc", "text": ""}, "text must not be empty."),
    ({"document_id": "doc", "text": "   \n\t "}, "text must not be empty."),
    ({"document_id": "", "text": "hello"}, "document_id must not be empty."),
    ({"document_id": "  ", "text": "hello"}, "document_id must not be empty."),
])
def test_empty_input_is_400(fake_pinecone, body, detail):
    r = client.post("/ingest", headers=AUTH, json=body)

    assert r.status_code == 400
    assert r.json()["detail"] == detail
    fake_pinecone.embed.assert_not_called()


@pytest.mark.parametrize("body", [
    {"text": "hello"},                                                     # missing document_id
    {"document_id": "a#b", "text": "hello"},                               # "#" would break chunk ids
    {"document_id": "doc", "text": "hello", "metadata": {"source": "x"}},  # reserved key
    {"document_id": "doc", "text": "hello", "metadata": {"n": {"nested": 1}}},
    {"document_id": "doc", "text": "x" * 500_001},
])
def test_invalid_input_is_422(fake_pinecone, body):
    assert client.post("/ingest", headers=AUTH, json=body).status_code == 422
    fake_pinecone.embed.assert_not_called()


def test_ingest_needs_token(fake_pinecone):
    assert client.post("/ingest", json={"document_id": "doc", "text": "hello"}).status_code == 401
    fake_pinecone.embed.assert_not_called()


# --- Upstream errors ------------------------------------------------------------

@pytest.mark.parametrize("error, expected_status, detail", [
    (UnauthorizedException(status=401, reason="Unauthorized"), 500, "Server Pinecone key is invalid."),
    (PineconeApiException(status=429, reason="Too Many Requests"), 429, "Pinecone rate limited, try again shortly."),
    (PineconeApiException(status=500, reason="Internal Server Error"), 502, "Vector store request failed."),
    (urllib3.exceptions.MaxRetryError(None, "/embed"), 502, "Could not reach Pinecone."),
])
def test_embedding_errors_are_mapped(fake_pinecone, error, expected_status, detail):
    fake_pinecone.embed.side_effect = error

    r = client.post("/ingest", headers=AUTH, json={"document_id": "doc", "text": "hello"})

    assert r.status_code == expected_status
    assert r.json()["detail"] == detail


def test_pinecone_upsert_error_is_502(fake_pinecone):
    fake_pinecone.index.upsert.side_effect = PineconeApiException(status=400, reason="Bad Request")

    r = client.post("/ingest", headers=AUTH, json={"document_id": "doc", "text": "hello"})

    assert r.status_code == 502
    assert r.json()["detail"] == "Vector store request failed."


def test_missing_pinecone_key_is_503(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY")
    rag.pinecone_client.cache_clear()
    rag.pinecone_index.cache_clear()

    r = client.post("/ingest", headers=AUTH, json={"document_id": "doc", "text": "hello"})

    assert r.status_code == 503
    assert r.json()["detail"] == "PINECONE_API_KEY is not set."
