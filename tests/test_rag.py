"""Tests for rag.py and GET /debug/pinecone. Pinecone is faked, so these cost nothing."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pinecone.exceptions import NotFoundException, UnauthorizedException

import app
import rag

client = TestClient(app.app)
AUTH = {"X-App-Token": "test-token"}


@pytest.fixture(autouse=True)
def reset_rate_limit():
    app.limiter.reset()


# --- Embeddings, upsert, query -----------------------------------------------

def test_ingest_and_query_use_the_same_model_with_matching_input_types(fake_pinecone):
    fake_pinecone.index.query.return_value = SimpleNamespace(matches=[])

    rag.upsert([{"id": "a", "text": "hello"}])
    rag.query("hi")

    calls = fake_pinecone.embed.call_args_list
    assert {c.kwargs["model"] for c in calls} == {"llama-text-embed-v2"}
    assert [c.kwargs["parameters"]["input_type"] for c in calls] == ["passage", "query"]


def test_embed_batches_and_keeps_order(fake_pinecone):
    texts = ["x" * n for n in range(1, 251)]

    vectors, tokens = rag.embed(texts, input_type="passage")

    assert fake_pinecone.embed.call_count == 3  # 96 + 96 + 58
    assert [v[0] for v in vectors] == [float(n) for n in range(1, 251)]
    assert tokens == 750


def test_upsert_stores_text_in_metadata(fake_pinecone):
    tokens = rag.upsert([{"id": "doc1-0", "text": "Pinecone stores vectors.", "metadata": {"source": "notes.md"}}])

    vectors = fake_pinecone.index.upsert.call_args.kwargs["vectors"]
    assert vectors[0]["id"] == "doc1-0"
    assert len(vectors[0]["values"]) == 1024
    assert vectors[0]["metadata"] == {"source": "notes.md", "text": "Pinecone stores vectors."}
    assert tokens == 3


def test_upsert_nothing_makes_no_calls(fake_pinecone):
    assert rag.upsert([]) == 0
    fake_pinecone.embed.assert_not_called()
    fake_pinecone.index.upsert.assert_not_called()


def test_query_lifts_known_fields_out_of_metadata(fake_pinecone):
    fake_pinecone.index.query.return_value = SimpleNamespace(matches=[
        SimpleNamespace(id="doc1#0", score=0.91, metadata={
            "text": "Chunk text", "document_id": "doc1", "chunk_index": 0.0, "source": "notes.md", "dept": "support",
        }),
        SimpleNamespace(id="loose", score=0.5, metadata={"text": "Upserted without document fields"}),
    ])

    matches, tokens = rag.query("What stores vectors?", top_k=3)

    assert matches == [
        {"id": "doc1#0", "score": 0.91, "text": "Chunk text", "document_id": "doc1", "chunk_index": 0,
         "source": "notes.md", "metadata": {"dept": "support"}},
        {"id": "loose", "score": 0.5, "text": "Upserted without document fields", "document_id": None,
         "chunk_index": None, "source": None, "metadata": {}},
    ]
    assert isinstance(matches[0]["chunk_index"], int)
    assert fake_pinecone.index.query.call_args.kwargs["top_k"] == 3
    assert tokens == 3


def test_embedding_cost():
    assert rag.embedding_cost(1_000_000) == pytest.approx(0.16)


# --- GET /debug/pinecone -----------------------------------------------------

def test_debug_pinecone_reports_index(fake_pinecone):
    r = client.get("/debug/pinecone", headers=AUTH)

    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "index": "test-index",
        "dimension": 1024,
        "metric": "cosine",
        "vector_count": 42,
        "embedding_model": "llama-text-embed-v2",
    }


def test_debug_pinecone_needs_token(fake_pinecone):
    assert client.get("/debug/pinecone").status_code == 401
    fake_pinecone.client.describe_index.assert_not_called()


def test_debug_pinecone_wrong_dimension_is_503(fake_pinecone):
    fake_pinecone.client.describe_index.return_value = SimpleNamespace(dimension=1536, metric="cosine")

    r = client.get("/debug/pinecone", headers=AUTH)

    assert r.status_code == 503
    assert "dimension 1536" in r.json()["detail"]


@pytest.mark.parametrize("error, detail", [
    (UnauthorizedException(status=401, reason="Unauthorized"), "Pinecone rejected PINECONE_API_KEY."),
    (NotFoundException(status=404, reason="Not Found"), "Pinecone index 'test-index' does not exist."),
    (ConnectionError("network down"), "Could not reach Pinecone."),
])
def test_debug_pinecone_errors_are_503(fake_pinecone, error, detail):
    fake_pinecone.client.describe_index.side_effect = error

    r = client.get("/debug/pinecone", headers=AUTH)

    assert r.status_code == 503
    assert r.json()["detail"] == detail


def test_debug_pinecone_missing_config_is_503(monkeypatch):
    monkeypatch.delenv("PINECONE_INDEX_NAME")

    r = client.get("/debug/pinecone", headers=AUTH)

    assert r.status_code == 503
    assert r.json()["detail"] == "PINECONE_INDEX_NAME is not set."
