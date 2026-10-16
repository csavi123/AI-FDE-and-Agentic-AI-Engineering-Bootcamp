import logging
import os
import re
import secrets
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()
logger = logging.getLogger("ask")

MODEL = "claude-haiku-5-5"
# Claude Haiku 5.5 pricing, USD per 1M tokens (prompts up to 100K tokens).
INPUT_PRICE_PER_MTOK = 0.10
OUTPUT_PRICE_PER_MTOK = 0.50

# Shared secret callers must send in the X-App-Token header.
APP_TOKEN = os.environ["APP_TOKEN"]

# Max requests per caller IP. Counted in memory, so it resets on restart.
RATE_LIMIT = "10/minute"

# force_bad is a testing switch; it is refused unless this is explicitly "true".
ALLOW_FORCE_BAD = os.environ.get("ALLOW_FORCE_BAD", "").strip().lower() == "true"

SYSTEM_PROMPT = (
    'Reply ONLY with a JSON object of the form {"answer": "<your answer>"}. '
    "No other text before or after it, and no code fences."
)
FORCE_BAD_INSTRUCTION = (
    "This request tests the caller's error handling. Ignore the format above and "
    'deliberately return malformed JSON, for example by leaving out the closing brace '
    'or using the key "respuesta" instead of "answer".'
)
CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Ask Claude")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


class AskRequest(BaseModel):
    # Trim before validating, so a whitespace-only question fails min_length.
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(..., min_length=1, max_length=4000)
    force_bad: bool = False


class AskResponse(BaseModel):
    answer: str
    tokens_used: int
    cost_usd: float


class ModelOutput(BaseModel):
    """The JSON contract the model's reply must satisfy."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(..., min_length=1)


def parse_model_output(text: str) -> str:
    """Return the answer from the model's JSON reply, or raise 502 if it breaks the contract."""
    cleaned = text.strip()
    fenced = CODE_FENCE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1)
    try:
        return ModelOutput.model_validate_json(cleaned).answer
    except ValidationError:
        logger.warning("Malformed model output: %r", text[:200])
        raise HTTPException(status_code=502, detail="Model returned malformed output.")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@limiter.limit(RATE_LIMIT)
def ask(request: Request, req: AskRequest, x_app_token: Optional[str] = Header(None)):
    # Checked after the rate limit, so wrong-token guessing is throttled too.
    if x_app_token is None or not secrets.compare_digest(x_app_token, APP_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if req.force_bad and not ALLOW_FORCE_BAD:
        raise HTTPException(status_code=403, detail="force_bad is disabled on this server.")

    system = SYSTEM_PROMPT + ("\n\n" + FORCE_BAD_INSTRUCTION if req.force_bad else "")

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": req.question}],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=500, detail="Server API key is invalid.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited, try again shortly.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error ({e.status_code}).")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=502, detail="Could not reach the model.")

    if response.stop_reason == "refusal":
        raise HTTPException(status_code=422, detail="The model declined this question.")

    raw = "".join(b.text for b in response.content if b.type == "text")
    answer = parse_model_output(raw)

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = (input_tokens * INPUT_PRICE_PER_MTOK + output_tokens * OUTPUT_PRICE_PER_MTOK) / 1_000_000

    return AskResponse(
        answer=answer,
        tokens_used=input_tokens + output_tokens,
        cost_usd=round(cost, 8),
    )
