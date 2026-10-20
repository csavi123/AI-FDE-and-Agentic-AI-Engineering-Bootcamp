"""Pinecone vector store and Pinecone-hosted embeddings for RAG.

Embeddings come from Pinecone Inference, so the one PINECONE_API_KEY covers both
embedding and storage. Every text goes through embed(), at ingest and at query time,
so stored vectors and question vectors always come from the same model and can be compared.
Clients are created on first use, so the app still starts (and /ask still works)
when the Pinecone settings are missing.
"""
import logging
import os
from functools import lru_cache
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pinecone.exceptions import NotFoundException, UnauthorizedException

load_dotenv()
logger = logging.getLogger("rag")

# Fixed in code, not configurable: changing it would make stored vectors unsearchable.
EMBEDDING_MODEL = "llama-text-embed-v2"
EMBEDDING_DIMENSION = 1024  # the model's default size; the Pinecone index must use the same
# llama-text-embed-v2 pricing, USD per 1M tokens, after the Starter plan's free monthly
# allowance. Check pinecone.io/pricing for current numbers.
EMBEDDING_PRICE_PER_MTOK = 0.16

EMBED_BATCH_SIZE = 96  # the model accepts at most 96 texts per request
# Vectors per Pinecone upsert: 1024 floats each as JSON is ~14KB, and a request is capped at 2MB.
UPSERT_BATCH_SIZE = 50


class VectorStoreError(Exception):
    """A configuration or connection problem with Pinecone, safe to show to callers."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise VectorStoreError(f"{name} is not set.")
    return value


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


# Chunk sizes are in characters. Read once at startup; a bad value stops the app from starting.
CHUNK_SIZE = _int_env("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int_env("CHUNK_OVERLAP", 100)
if CHUNK_SIZE <= 0 or not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(f"Need CHUNK_SIZE > 0 and 0 <= CHUNK_OVERLAP < CHUNK_SIZE, got {CHUNK_SIZE}/{CHUNK_OVERLAP}.")


@lru_cache(maxsize=1)
def pinecone_client() -> Pinecone:
    return Pinecone(api_key=_require("PINECONE_API_KEY"))


def index_name() -> str:
    return _require("PINECONE_INDEX_NAME")


@lru_cache(maxsize=1)
def pinecone_index():
    return pinecone_client().Index(index_name())


def embed(texts: List[str], input_type: str) -> Tuple[List[List[float]], int]:
    """Return one vector per text, in order, and the embedding tokens used.

    input_type is "passage" for stored chunks and "query" for questions; the model
    encodes the two slightly differently so questions land near the passages that answer them.
    """
    vectors, tokens = [], 0
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        response = pinecone_client().inference.embed(
            model=EMBEDDING_MODEL,
            inputs=texts[start:start + EMBED_BATCH_SIZE],
            parameters={"input_type": input_type, "truncate": "END"},
        )
        vectors.extend(item["values"] for item in response.data)
        tokens += response.usage["total_tokens"]
    return vectors, tokens


def upsert(records: List[dict]) -> int:
    """Embed and store records of the form {"id", "text", "metadata"?}. Returns embedding tokens used.

    The text is stored in the metadata so a search can hand it straight to the model.
    Upserting an existing id overwrites it.
    """
    if not records:
        return 0
    vectors, tokens = embed([r["text"] for r in records], input_type="passage")
    pinecone_index().upsert(
        vectors=[
            {"id": r["id"], "values": v, "metadata": {**r.get("metadata", {}), "text": r["text"]}}
            for r, v in zip(records, vectors)
        ],
        batch_size=UPSERT_BATCH_SIZE,
        show_progress=False,
    )
    return tokens


def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return [c for c in splitter.split_text(text) if c.strip()]


def chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}#{chunk_index}"


def index_document(document_id: str, text: str, source: str, metadata: Optional[dict] = None) -> Tuple[int, int]:
    """Chunk, embed and store one document. Returns (chunks indexed, embedding tokens used).

    Re-indexing a document_id replaces it: chunks are upserted under the same ids, then
    any leftover chunks from a longer earlier version are deleted.
    """
    chunks = split_text(text)
    records = [
        {
            "id": chunk_id(document_id, i),
            "text": chunk,
            "metadata": {**(metadata or {}), "document_id": document_id, "chunk_index": i, "source": source},
        }
        for i, chunk in enumerate(chunks)
    ]
    tokens = upsert(records)

    current = {r["id"] for r in records}
    for ids in pinecone_index().list(prefix=f"{document_id}#"):
        stale = [i for i in ids if i not in current]
        if stale:
            pinecone_index().delete(ids=stale)
    return len(records), tokens


def query(question: str, top_k: int = 5, filter: Optional[dict] = None) -> Tuple[List[dict], int]:
    """Return the top_k most similar stored chunks, best first, and the embedding tokens used.

    Each match has id, score (cosine similarity, higher is closer), text, document_id,
    chunk_index, source, and any other metadata under "metadata".
    """
    [vector], tokens = embed([question], input_type="query")
    result = pinecone_index().query(vector=vector, top_k=top_k, filter=filter, include_metadata=True)
    matches = []
    for m in result.matches:
        metadata = dict(m.metadata or {})
        chunk_index = metadata.pop("chunk_index", None)
        matches.append({
            "id": m.id,
            "score": m.score,
            "text": metadata.pop("text", ""),
            "document_id": metadata.pop("document_id", None),
            # Pinecone returns every number as a float; chunk_index is a position, so make it an int again.
            "chunk_index": int(chunk_index) if chunk_index is not None else None,
            "source": metadata.pop("source", None),
            "metadata": metadata,
        })
    return matches, tokens


def embedding_cost(tokens: int) -> float:
    return tokens * EMBEDDING_PRICE_PER_MTOK / 1_000_000


def pinecone_health() -> dict:
    """Confirm Pinecone is reachable and the index matches the embedding model. Costs nothing.

    Raises VectorStoreError with a readable reason when something is wrong.
    """
    name = index_name()
    try:
        description = pinecone_client().describe_index(name)
        stats = pinecone_index().describe_index_stats()
    except UnauthorizedException:
        raise VectorStoreError("Pinecone rejected PINECONE_API_KEY.")
    except NotFoundException:
        raise VectorStoreError(f"Pinecone index '{name}' does not exist.")

    if description.dimension != EMBEDDING_DIMENSION:
        raise VectorStoreError(
            f"Index '{name}' has dimension {description.dimension}, but {EMBEDDING_MODEL} "
            f"needs {EMBEDDING_DIMENSION}. Recreate the index with dimension {EMBEDDING_DIMENSION}."
        )
    return {
        "status": "ok",
        "index": name,
        "dimension": description.dimension,
        "metric": description.metric,
        "vector_count": stats.total_vector_count,
        "embedding_model": EMBEDDING_MODEL,
    }
