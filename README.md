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

## Architecture

```
Browser ──► Streamlit server ──HTTPS + X-App-Token──► FastAPI /ask ──► Anthropic API (Claude Haiku 5.5)
            (streamlit_app.py)                        (app.py)
            holds APP_TOKEN                           holds ANTHROPIC_API_KEY
```

Each secret lives in exactly one place. The browser never sees either one.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | FastAPI backend: `GET /` (health check) and `POST /ask` |
| `streamlit_app.py` | One-page UI that calls `/ask` from the server side |
| `.streamlit/config.toml` | Dark theme for the UI |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for local secrets (copy to `.env`) |
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

`.env` is git-ignored. Never commit it.

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

### `GET /`

Health check. Returns `{"status": "ok"}`. No auth required.

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

## How cost is calculated

```
cost_usd = (input_tokens × $0.10 + output_tokens × $0.50) / 1,000,000
```

These are Claude Haiku 5.5's prices for prompts up to 100K tokens. `tokens_used` includes the model's hidden thinking tokens, which are billed as output, so it can be larger than the visible answer suggests.

## Security and cost controls

- **Shared-secret auth:** `/ask` requires `X-App-Token`, compared in constant time.
- **Rate limiting:** 10 requests per minute per IP, using `slowapi`. It's checked before auth, so token guessing is throttled too.
- **Input validation:** empty, whitespace-only, and over-4000-character questions are rejected before Claude is called, so they cost nothing.
- **Output cap:** `max_tokens=4096` limits the length of each response.
- **Spend limit (recommended):** set a monthly limit in the Anthropic Console under Settings → Limits as a hard backstop.

## Deploy to Render (public HTTPS URL)

1. Push this repo to GitHub. `.env` stays local.
2. On [render.com](https://render.com), choose **New → Blueprint** and select the repo. Render reads `render.yaml`.
3. Enter `ANTHROPIC_API_KEY` and `APP_TOKEN` when prompted.
4. When the deploy shows **Live**, call `https://<your-service>.onrender.com/ask` with the same `curl` command.

To point the local UI at the deployed API, set `API_URL=https://<your-service>.onrender.com` in `.env`.

> On Render's free plan, the service sleeps when idle. The first request after a while can take 30–60 seconds.

## Known limitations and next steps

- The rate limiter is in memory: it resets on restart and isn't shared across instances. Redis would fix this.
- There's one shared token rather than per-user API keys or OAuth.
- No streaming: the answer appears all at once.
- No structured logging or metrics.
- The UI is not deployed. It runs locally only.
