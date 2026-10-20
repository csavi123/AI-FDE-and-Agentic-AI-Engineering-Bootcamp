"""Golden-set eval: runs eval/golden_set.json against a running API and prints a markdown table.

Usage:
    python eval/run_eval.py                      # API_URL from .env, else http://127.0.0.1:8000
    python eval/run_eval.py --api-url https://<your-service>.onrender.com

Calls the live API (real Claude and Pinecone), so it costs about $0.0002 per question.
Retrieval hit, correctness and refusal are checked automatically. Faithfulness is not:
the script prints each answer next to the chunk that should support it, for you to judge.
Exits 1 if any automatic check fails.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

REFUSAL = "I don't have enough information to answer that."
TOP_K = 5  # matches RETRIEVAL_TOP_K in app.py
SEARCH_K = 20  # deeper search, only to report the answer chunk's rank on a miss
PAUSE_SECONDS = 7  # two calls per question; stays under the API's 10 requests per minute
GOLDEN_SET = Path(__file__).parent / "golden_set.json"


def normalize(text):
    return " ".join(text.split()).lower()


def call(session, method, url, **kwargs):
    """One API call, retried once after a minute if rate limited."""
    for attempt in range(2):
        response = session.request(method, url, timeout=120, **kwargs)
        if response.status_code == 429 and attempt == 0:
            print("  rate limited, waiting 60 s...", file=sys.stderr)
            time.sleep(60)
            continue
        response.raise_for_status()
        time.sleep(PAUSE_SECONDS)
        return response.json()


def evaluate(session, api_url, case):
    retrieved = call(session, "GET", f"{api_url}/debug/retrieve",
                     params={"q": case["question"], "top_k": SEARCH_K})
    asked = call(session, "POST", f"{api_url}/ask", json={"question": case["question"]})

    answer = asked["answer"]
    result = {"case": case, "answer": answer, "citations": asked["citations"],
              "cost_usd": asked["cost_usd"], "answer_chunks": [], "rank": None}

    if case["expected_document"] is None:
        result["refused"] = answer.strip() == REFUSAL and not asked["citations"]
        return result

    # The answer chunk is any chunk containing the answer phrase. Matching on text rather than
    # fixed chunk ids keeps the golden set valid when chunk size or overlap changes.
    phrase = normalize(case["answer_phrase"])
    matches = retrieved["matches"]
    answer_chunks = [m for m in matches if phrase in normalize(m["text"])]
    answer_ids = {m["id"] for m in answer_chunks}
    result["answer_chunks"] = answer_chunks
    result["rank"] = next((i + 1 for i, m in enumerate(matches) if m["id"] in answer_ids), None)

    # Hit = Claude was actually given an answer chunk.
    result["retrieval_hit"] = bool(answer_ids & set(asked["retrieved_chunk_ids"][:TOP_K]))
    result["cites_expected"] = case["expected_document"] in asked["citations"]
    lowered = answer.lower()
    result["correct"] = (answer.strip() != REFUSAL and all(
        any(variant.lower() in lowered for variant in group)
        for group in case["answer_must_include"]))
    return result


def mark(ok):
    return "✅" if ok else "❌"


def cell(text):
    return text.replace("|", "\\|").replace("\n", " ")


def print_table(results):
    print("| # | Question | Expected answer | Answer chunk | Rank | Retrieval hit (top-5) "
          "| Cites expected doc | Correct | Faithful |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        case = r["case"]
        if case["expected_document"] is None:
            print(f"| {case['id']} | **Refusal:** {cell(case['question'])} | \"{REFUSAL}\" | none "
                  f"| n/a | n/a | n/a | {mark(r['refused'])} | {mark(r['refused'])} |")
            continue
        chunks = ", ".join(f"`{m['id']}`" for m in r["answer_chunks"]) or "not in top 20"
        print(f"| {case['id']} | {cell(case['question'])} | {cell(case['expected_answer'])} "
              f"| {chunks} | {r['rank'] or '>20'} | {mark(r['retrieval_hit'])} "
              f"| {mark(r['cites_expected'])} | {mark(r['correct'])} | ☐ review |")


def print_review(results):
    print("\n## Faithfulness review\n")
    print("For each row, check that every fact in the answer appears in the chunk text.\n")
    for r in results:
        case = r["case"]
        print(f"### {case['id']}. {case['question']}\n")
        print(f"- **Expected:** {case['expected_answer']}")
        print(f"- **Answer:** {r['answer']}")
        print(f"- **Citations:** {r['citations']}\n")
        for m in r["answer_chunks"]:
            print(f"`{m['id']}` (score {m['score']:.3f}):\n\n```\n{m['text'].strip()}\n```\n")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-url",
                        default=os.environ.get("API_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()
    api_url = args.api_url.rstrip("/")

    token = os.environ.get("APP_TOKEN")
    if not token:
        sys.exit("APP_TOKEN is not set. Add it to .env or the environment.")

    cases = json.loads(GOLDEN_SET.read_text())
    session = requests.Session()
    session.headers["X-App-Token"] = token

    results = []
    for case in cases:
        print(f"[{case['id']}/{len(cases)}] {case['question']}", file=sys.stderr)
        results.append(evaluate(session, api_url, case))

    answerable = [r for r in results if r["case"]["expected_document"] is not None]
    refusals = [r for r in results if r["case"]["expected_document"] is None]
    hits = sum(r["retrieval_hit"] for r in answerable)
    cited = sum(r["cites_expected"] for r in answerable)
    correct = sum(r["correct"] for r in answerable)
    refused = sum(r["refused"] for r in refusals)
    cost = sum(r["cost_usd"] for r in results)

    print_table(results)
    print(f"\n**Automatic checks:** retrieval {hits}/{len(answerable)} · "
          f"cites expected doc {cited}/{len(answerable)} · "
          f"correct {correct}/{len(answerable)} · refusal {refused}/{len(refusals)} · "
          f"cost ${cost:.4f}")
    print_review(results)

    failed = (hits < len(answerable) or cited < len(answerable)
              or correct < len(answerable) or refused < len(refusals))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
