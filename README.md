# Ask Claude

A small question-answering service. Send a question to `POST /ask` and get back a clean, predictable JSON response from **Claude Haiku 5.5**:

```json
{
  "answer": "RAG (Retrieval-Augmented Generation) is ...",
  "tokens_used": 142,
  "cost_usd": 0.0000521
}
```

It ships with a one-page Streamlit UI and is deployable to Render for a public HTTPS URL.

**Session 2 (RAG, in progress):** documents can be stored in a **Pinecone** vector store with `POST /ingest` and searched with `GET /debug/retrieve`. `/ask` doesn't use them yet: it still answers from Claude's own knowledge. Connecting retrieval to `/ask` is the next step.

## Architecture

```
Browser ──► Streamlit server ──HTTPS + X-App-Token──► FastAPI /ask/stream ──► Anthropic API (Claude Haiku 5.5)
            (streamlit_app.py)                        (app.py)
            holds APP_TOKEN                           holds ANTHROPIC_API_KEY
```

The RAG side, also in `app.py`, with the work done in `rag.py`:

```
POST /ingest          text ──► chunk ──► embed ("passage") ──► upsert ──► Pinecone index
GET  /debug/retrieve  question ──► embed ("query") ──► top-k similarity search ──► chunks + scores (no Claude)
                                   └─ llama-text-embed-v2, hosted by Pinecone (holds PINECONE_API_KEY)
```

Each secret lives in exactly one place. The browser never sees any of them.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | FastAPI backend: `GET /` (health check), `POST /ask`, `POST /ask/stream`, `POST /ingest`, `GET /debug/retrieve`, `GET /debug/pinecone` |
| `rag.py` | RAG building blocks: chunking, embeddings, Pinecone upsert and search, health check |
| `streamlit_app.py` | One-page UI that calls `/ask/stream` from the server side |
| `.streamlit/config.toml` | Dark theme for the UI |
| `tests/` | pytest suite (`test_ask.py`, `test_ingest.py`, `test_retrieve.py`, `test_rag.py`) and shared fakes and test setup (`conftest.py`) |
| `pytest.ini` | pytest settings |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for local secrets (copy to `.env`) |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Run the API and UI locally in containers |
| `render.yaml` | Render deployment blueprint |

## Quick start (local)

Requires Python 3.9+.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your local secrets file
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your key from [console.anthropic.com](https://console.anthropic.com) |
| `APP_TOKEN` | A password of your choice. Callers must send it in the `X-App-Token` header. Generate one with `python3 -c "import secrets; print(secrets.token_urlsafe(24))"` |
| `API_URL` | *(optional, UI only)* Where the UI sends questions. Defaults to `http://127.0.0.1:8000` |
| `ALLOW_FORCE_BAD` | *(optional)* `true` allows the `force_bad` test switch and shows the "Simulate bad output" toggle in the UI. Leave it unset or `false` anywhere public. |
| `PINECONE_API_KEY` | *(RAG)* Your key from [app.pinecone.io](https://app.pinecone.io) → API Keys. Used for both embeddings and storage. |
| `PINECONE_INDEX_NAME` | *(RAG)* The name of your Pinecone index, e.g. `ask-rag`. See below. |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | *(optional)* Chunk size and overlap for `/ingest`, in **characters**. Defaults are `800` and `100`. Overlap must be smaller than size. |

`.env` is git-ignored. Never commit it. The app reads `.env` once at startup, so restart `uvicorn` after you change it.

### Create the Pinecone index (RAG only)

In the Pinecone console, create an index with **Custom settings** (not a model preset):

| Setting | Value |
|---|---|
| Name | the same as `PINECONE_INDEX_NAME`, e.g. `ask-rag` |
| Dimension | **1024** (the size of `llama-text-embed-v2` vectors) |
| Metric | **cosine** |
| Type | Serverless, dense, any cloud or region |

`GET /debug/pinecone` tells you if the dimension is wrong. Without these variables the RAG endpoints return `503`, but `/ask` keeps working.

### Run the API

```bash
uvicorn app:app --reload
```

### Call it

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "X-App-Token: YOUR_APP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is RAG in one sentence?"}'
```

Interactive API docs are at http://127.0.0.1:8000/docs.

### Run the UI

In a second terminal, with the virtual environment activated:

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501. Type a question and click **↑**, or press **⌘/Ctrl + Enter**.

The UI uses `/ask/stream`, so the answer appears word by word. When the page opens, it checks whether the API is awake. If the API is asleep (Render's free plan sleeps after 15 idle minutes), the page shows a short "Waking up the server…" note and starts waking it in the background while you type.

## API reference

### `POST /ask`

**Headers:** `X-App-Token: <APP_TOKEN>`, `Content-Type: application/json`

**Body:**
```json
{ "question": "string, 1 to 4000 characters", "force_bad": false }
```

`force_bad` is optional and only for testing. See [Output guardrail and `force_bad`](#output-guardrail-and-force_bad).

**Responses:**

| Status | Meaning |
|---|---|
| `200` | `{"answer": str, "tokens_used": int, "cost_usd": float}` |
| `401` | Missing or wrong `X-App-Token` |
| `403` | `force_bad` was sent but the server has `ALLOW_FORCE_BAD` off |
| `422` | Invalid body (empty or too long), or the model declined the question |
| `429` | Rate limit exceeded (10 requests per minute per IP), or Anthropic rate limit |
| `500` | Server's Anthropic API key is invalid |
| `502` | Anthropic API error or unreachable, or the model's reply broke the JSON contract (`"Model returned malformed output."`) |

### `POST /ask/stream`

Same headers, body and checks as `/ask`, but the answer is delivered as [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events):

```
event: delta
data: {"text": "RAG is "}

event: delta
data: {"text": "retrieval plus generation."}

event: done
data: {"answer": "RAG is retrieval plus generation.", "tokens_used": 113, "cost_usd": 0.000036}
```

- `delta` events carry the answer text as it is generated.
- The stream always ends with exactly one `done` event (the validated `/ask` response) or one `error` event, such as `{"status": 502, "detail": "Model returned malformed output."}`.
- The output guardrail can only validate the reply once it's complete. **When a client receives `error`, it must discard any text it has already shown.** The UI does this.
- Errors that happen before streaming starts (401, 403, 422, 429, or failure to reach the model) are returned as normal HTTP errors, the same as `/ask`.

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "X-App-Token: YOUR_APP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is RAG in one sentence?"}'
```

### `GET /`

Health check. Returns `{"status": "ok"}`. No auth required.

### `POST /ingest`

Chunks a document, embeds each chunk and stores it in Pinecone. Claude is not involved.

**Headers:** `X-App-Token: <APP_TOKEN>`, `Content-Type: application/json`

**Body:**
```json
{
  "document_id": "POL-101",
  "text": "the full document text, up to 500,000 characters",
  "source": "doc1_handbook.txt",
  "metadata": {"department": "hr"}
}
```

- `document_id` (required): letters, digits, `.`, `_`, `:` and `-`, up to 200 characters. Chunks are stored as `<document_id>#<chunk_index>`, e.g. `POL-101#0`.
- `text` (required): the document's content. The endpoint reads text, not files (see below for sending a file).
- `source` (optional): where the text came from, e.g. a file name. Defaults to `document_id`.
- `metadata` (optional): extra labels stored with every chunk. Values can be strings, numbers, booleans or lists of strings. `document_id`, `chunk_index`, `source` and `text` are reserved.

**What happens:** the text is split with LangChain's `RecursiveCharacterTextSplitter` (default 800 characters with 100 overlap). Each chunk is embedded with `llama-text-embed-v2` and upserted with metadata `document_id`, `chunk_index`, `source`, `text` and your extra labels. **Sending the same `document_id` again replaces that document:** its chunks are overwritten, and leftover chunks from a longer earlier version are deleted.

**Responses:**

| Status | Meaning |
|---|---|
| `200` | `{"document_id": str, "chunks_indexed": int, "status": "indexed", "tokens_used": int, "cost_usd": float}` |
| `400` | `text` or `document_id` is empty or whitespace-only |
| `401` | Missing or wrong `X-App-Token` |
| `422` | Invalid body: missing field, `#` or other disallowed characters in `document_id`, reserved or nested metadata, text too long |
| `429` | Rate limit exceeded, or Pinecone rate limit |
| `500` | Server's Pinecone API key is invalid |
| `502` | Pinecone error or unreachable |
| `503` | Pinecone isn't configured (`PINECONE_API_KEY` or `PINECONE_INDEX_NAME` not set) |

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "X-App-Token: YOUR_APP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "handbook", "text": "Remote work: up to 3 days per week."}'
```

**To ingest a file**, read it into the JSON body. Python handles the quotes and line breaks inside the file:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "X-App-Token: YOUR_APP_TOKEN" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"document_id": sys.argv[1], "source": sys.argv[2], "text": open(sys.argv[3]).read()}))' POL-101 doc1_handbook.txt path/to/doc1_handbook.txt)"
```

### `GET /debug/retrieve`

Shows which chunks a question retrieves, ranked by similarity. **It never calls Claude**, so you can check retrieval on its own before trusting a generated answer.

**Headers:** `X-App-Token: <APP_TOKEN>`

**Query parameters:** `q` (required, up to 4000 characters) and `top_k` (optional, 1 to 20, default 5).

```bash
curl -G http://127.0.0.1:8000/debug/retrieve \
  -H "X-App-Token: YOUR_APP_TOKEN" \
  --data-urlencode "q=How many remote days are allowed?"
```

```json
{
  "query": "How many remote days are allowed?",
  "top_k": 5,
  "matches": [
    {"id": "handbook#0", "score": 0.67, "text": "Remote work: up to 3 days per week.",
     "document_id": "handbook", "chunk_index": 0, "source": "handbook", "metadata": {}}
  ],
  "tokens_used": 9,
  "cost_usd": 0.00000144
}
```

`score` is cosine similarity: higher means closer in meaning. Search always returns the nearest chunks, even when none of them is relevant, so a low top score means the documents probably don't cover the question. Errors: `400` for an empty `q`, `401` for a missing or wrong token, `422` for `top_k` outside 1–20, and the same `429`/`500`/`502`/`503` as `/ingest`.

### `GET /debug/pinecone`

Checks that Pinecone is reachable and that the index matches the embedding model. It costs nothing.

**Headers:** `X-App-Token: <APP_TOKEN>`

```json
{"status": "ok", "index": "ask-rag", "dimension": 1024, "metric": "cosine",
 "vector_count": 12, "embedding_model": "llama-text-embed-v2"}
```

When something is wrong, it returns `503` with a plain reason, for example `"PINECONE_INDEX_NAME is not set."`, `"Pinecone rejected PINECONE_API_KEY."`, `"Pinecone index 'x' does not exist."` or a dimension mismatch. `vector_count` can take a few seconds to update after an ingest.

## Output guardrail and `force_bad`

The model is told to reply with JSON in the form `{"answer": "..."}`. Before anything is returned, `parse_model_output()` in `app.py` validates that reply against a `ModelOutput` schema:

- It must be valid JSON.
- It must have exactly one key, `answer`, and the answer can't be empty.
- A surrounding ```` ```json ```` code fence is tolerated.

**When validation fails,** the server logs a warning with the first 200 characters of the reply and returns a controlled error: `502 {"detail": "Model returned malformed output."}`. There is no retry and no fallback answer, so a broken reply never reaches the user. The model call has already happened by that point, so its tokens are still billed.

**`force_bad`** proves the guardrail works. When it's `true`, the system prompt tells the model to deliberately return malformed JSON, and the guardrail should reject it. It's refused with `403` unless the server has `ALLOW_FORCE_BAD=true`. With that flag on, the UI shows a **Simulate bad output** toggle. The model doesn't always obey the instruction, so a live `force_bad` request can occasionally succeed. The tests use a fake model that always returns broken output.

In production, structured outputs (`output_config.format`) would guarantee schema-valid JSON from the API. This validator would then remain as defense in depth.

## Tests

```bash
pytest
```

The tests in `tests/test_ask.py` replace Claude with a fake, so they need no API key, never call Anthropic, and cost nothing. They cover:

- **Happy path:** the response shape, the cost calculation, and that the question is trimmed before being sent
- **Validation (422):** missing, empty, whitespace-only, too long, or wrong type. Each test also checks that Claude was never called.
- **Auth (401):** missing or wrong `X-App-Token`
- **Rate limiting (429):** the 11th request in a minute is blocked, including token guesses
- **Refusals and upstream errors:** mapped to 422, 500, 429 and 502
- **Output guardrail (502):** missing brace, wrong key, empty answer, extra field, or plain text from the model; code-fenced JSON is accepted
- **`force_bad`:** refused with 403 when disabled; when enabled, the instruction is added and the broken reply is caught
- **Health check:** works without a token
- **Streaming:** deltas arrive first, then a validated `done`. Malformed, refused or interrupted streams end with an `error` event. Errors before streaming starts are plain HTTP errors. The incremental JSON decoder handles partial escapes and emoji.

The RAG tests replace Pinecone with a fake (in `tests/conftest.py`), so they need no Pinecone key either:

- **`tests/test_rag.py`:** ingest and search use the same embedding model, with `passage` and `query` input types; embedding batches stay in order; chunk text is stored in metadata; search results expose `document_id`, `chunk_index` and `source`; the embedding cost; and every `/debug/pinecone` outcome (ok, bad key, missing index, wrong dimension, missing setting, unreachable).
- **`tests/test_ingest.py`:** short and long documents, chunk ids and metadata, configurable chunk size, re-ingesting a shorter version deletes leftover chunks, `400` for empty input, `422` for invalid input, `401` without a token, and Pinecone errors mapped to `500`, `429` and `502`. Bad requests never reach the embedding model.
- **`tests/test_retrieve.py`:** ranked matches with scores and document ids, the question embedded as a query, `top_k` default and limits, bad parameters, auth, and that Claude is never called.

## How cost is calculated

```
cost_usd = (input_tokens × $0.10 + output_tokens × $0.50) / 1,000,000
```

These are Claude Haiku 5.5's prices for prompts up to 100K tokens. `tokens_used` includes the model's hidden thinking tokens, which are billed as output, so it can be larger than the visible answer suggests.

For `/ingest` and `/debug/retrieve`, `tokens_used` and `cost_usd` cover embedding only:

```
cost_usd = embedding_tokens × $0.16 / 1,000,000
```

That's the price of `llama-text-embed-v2` on Pinecone beyond the Starter plan's free monthly allowance, so on the free plan the real charge is usually $0. Check [pinecone.io/pricing](https://www.pinecone.io/pricing/) for current numbers. Storage and search themselves aren't included.

## Security and cost controls

- **Shared-secret auth:** every endpoint except `GET /` requires `X-App-Token`, compared in constant time. That includes the RAG and debug endpoints, so nobody else can fill or read your vector store.
- **Rate limiting:** 10 requests per minute per IP on each endpoint, using `slowapi`. It's checked before auth, so token guessing is throttled too.
- **Input validation:** empty, whitespace-only, and over-4000-character questions are rejected before Claude is called, so they cost nothing. `/ingest` and `/debug/retrieve` likewise reject bad input before anything is embedded.
- **Output cap:** `max_tokens=4096` limits the length of each response.
- **Spend limit (recommended):** set a monthly limit in the Anthropic Console under Settings → Limits as a hard backstop.

## Deploy to Render (public HTTPS URL)

1. Push this repo to GitHub. `.env` stays local.
2. On [render.com](https://render.com), choose **New → Blueprint** and select the repo. Render reads `render.yaml`.
3. Enter `ANTHROPIC_API_KEY`, `APP_TOKEN`, `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` when prompted. For an existing service, add the two Pinecone variables under **Environment**.
4. When the deploy shows **Live**, call `https://<your-service>.onrender.com/ask` with the same `curl` command, and `GET /debug/pinecone` to confirm Pinecone is connected.

The vectors live in Pinecone, not on Render's disk, so they survive Render restarts and redeploys. The local app and the deployed app share the same index when they use the same `PINECONE_INDEX_NAME`.

To point the local UI at the deployed API, set `API_URL=https://<your-service>.onrender.com` in `.env`.

> On Render's free plan, the service sleeps when idle. The first request after a while can take 30–60 seconds. The UI shows a "Waking up the server…" note while this happens.

## Known limitations and next steps

- The rate limiter is in memory: it resets on restart and isn't shared across instances. Redis would fix this.
- There's one shared token rather than per-user API keys or OAuth.
- With streaming, a malformed reply is visible for a moment before the UI removes it, because it can only be validated once complete.
- When the UI is hosted, every user's request reaches the API from the UI server's IP, so all UI users share one 10-per-minute limit.
- No structured logging or metrics.
- **`/ask` doesn't use the vector store yet.** Retrieval works (`/debug/retrieve`) but isn't connected to generation.
- **Embedding model:** the assignment suggested OpenAI's `text-embedding-3-small`. This project uses Pinecone-hosted `llama-text-embed-v2` instead, so one Pinecone key covers both embeddings and storage and no OpenAI account is needed. The model is fixed in `rag.py` on purpose: changing it means recreating the index and re-ingesting every document, because vectors from different models can't be compared.
- `/ingest` takes text in JSON. There's no file upload yet.
- Chunk sizes are measured in characters, not tokens.
- Everything goes into the index's default namespace, so all documents are searched together.
