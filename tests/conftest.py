import os

# Set before app.py is imported. load_dotenv() never overrides existing variables,
# so tests use these dummies instead of the real secrets in .env.
os.environ["APP_TOKEN"] = "test-token"
os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["ALLOW_FORCE_BAD"] = "false"  # tests turn it on explicitly where needed
os.environ["PINECONE_API_KEY"] = "test-pinecone-key"
os.environ["PINECONE_INDEX_NAME"] = "test-index"

from types import SimpleNamespace  # noqa: E402  (imports must follow the env setup above)
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402

import rag  # noqa: E402


@pytest.fixture
def fake_pinecone(monkeypatch):
    """A Pinecone client and index that store nothing; set index.list.return_value to fake existing ids.

    Embedding (client.inference.embed) returns [len(text), 0.0, ...] per text and 3 tokens per text.
    """
    pc, index = MagicMock(), MagicMock()

    def embed(model, inputs, parameters):
        return SimpleNamespace(
            data=[{"values": [float(len(t))] + [0.0] * (rag.EMBEDDING_DIMENSION - 1)} for t in inputs],
            usage={"total_tokens": 3 * len(inputs)},
        )

    pc.inference.embed.side_effect = embed
    pc.describe_index.return_value = SimpleNamespace(dimension=rag.EMBEDDING_DIMENSION, metric="cosine")
    index.describe_index_stats.return_value = SimpleNamespace(total_vector_count=42)
    index.list.return_value = iter([])
    monkeypatch.setattr(rag, "pinecone_client", lambda: pc)
    monkeypatch.setattr(rag, "pinecone_index", lambda: index)
    return SimpleNamespace(client=pc, index=index, embed=pc.inference.embed)
