import html
import json
import os
import threading

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# The UI calls the API from the server side, so APP_TOKEN never reaches the browser.
# This is only a thin client: retrieval, prompting and validation all happen in the API.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
APP_TOKEN = os.environ.get("APP_TOKEN", "")
# Shows the "Simulate bad output" toggle; keep it off anywhere public.
ALLOW_FORCE_BAD = os.environ.get("ALLOW_FORCE_BAD", "").strip().lower() == "true"

st.set_page_config(page_title="Ask Claude", page_icon="✦", layout="centered")

# Prompt-box look inspired by 21st.dev's "AI prompt box": sunset gradient page,
# dark rounded input, round white button.
st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(ellipse 80% 70% at 50% 108%, #E2592A 0%, #E46A2C 22%, #E8983A 40%, rgba(232, 175, 140, 0.7) 58%, rgba(228, 176, 205, 0) 78%),
          linear-gradient(180deg, #A9C3E3 0%, #C3B4D9 30%, #DDB1D2 60%, #E4B3C9 100%);
      }
      [data-testid="stHeader"] { background: transparent; }
      .block-container { max-width: 760px; padding-top: 14vh; }

      /* Tabs: a dark pill so the labels stay readable on the gradient */
      [data-baseweb="tab-list"] {
        background: rgba(31, 32, 35, 0.85); border-radius: 999px; padding: 0.25rem 0.5rem;
        width: fit-content; margin-bottom: 0.75rem;
      }
      [data-baseweb="tab"] { padding: 0.35rem 0.9rem; }
      [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }

      /* The prompt box */
      [data-testid="stForm"] {
        background: #1F2023;
        border: 1px solid #444444;
        border-radius: 28px;
        padding: 0.6rem 0.9rem 0.6rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.28);
      }
      [data-testid="stForm"] [data-baseweb="textarea"],
      [data-testid="stForm"] [data-baseweb="base-input"],
      [data-testid="stForm"] textarea {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
      }
      [data-testid="stForm"] textarea {
        color: #F3F4F6 !important;
        -webkit-text-fill-color: #F3F4F6 !important;
        caret-color: #F3F4F6;
        font-size: 1.1rem;
        padding: 0.5rem 0.25rem;
        resize: none;
      }
      [data-testid="stForm"] textarea::placeholder {
        color: #9CA3AF !important;
        -webkit-text-fill-color: #9CA3AF !important;
      }
      [data-testid="stForm"] [data-testid="InputInstructions"] { display: none; }


      /* Round white send button, pinned to the right edge */
      [data-testid="stForm"] [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"] { align-items: flex-end; }
      [data-testid="stFormSubmitButton"] { display: flex; justify-content: flex-end; }
      [data-testid="stFormSubmitButton"] button {
        width: 2.75rem; height: 2.75rem; min-height: 0; padding: 0;
        border-radius: 999px; border: none;
        background: #FFFFFF; color: #1F2023;
      }
      [data-testid="stFormSubmitButton"] button p { font-size: 1.3rem; font-weight: 700; color: #1F2023; }
      [data-testid="stFormSubmitButton"] button:hover { background: #E5E7EB; }

      /* "Simulate bad output" toggle */
      [data-testid="stForm"] [data-testid="stCheckbox"] label p { color: #9CA3AF; font-size: 0.85rem; }

      /* The "Add document" panel: same dark card as the prompt box */
      .st-key-ingest_panel {
        background: #1F2023; border: 1px solid #444444; border-radius: 28px;
        padding: 1.25rem 1.4rem; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.28);
      }

      /* Results */
      .answer-card {
        background: #1F2023; border: 1px solid #444444; border-radius: 24px;
        padding: 1.25rem 1.5rem; margin-top: 1rem; color: #E5E7EB; line-height: 1.6;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
      }
      .answer-card.refused { border-color: #D97706; }
      .answer-card.refused .label { color: #FBBF24; }
      .muted { color: #9CA3AF; }
      .cursor { color: #9B87F5; animation: blink 1s steps(1) infinite; }
      @keyframes blink { 50% { opacity: 0; } }
      .note {
        margin-top: 0.9rem; padding: 0.7rem 1.1rem; border-radius: 16px;
        background: rgba(31, 32, 35, 0.85); border: 1px solid #444444;
        color: #D1D5DB; font-size: 0.9rem; line-height: 1.5;
      }
      .label { color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
      .sources { margin-top: 0.9rem; padding-top: 0.8rem; border-top: 1px solid #333438; }
      .chip {
        display: inline-block; margin: 0.35rem 0.35rem 0 0; padding: 0.15rem 0.65rem;
        border-radius: 999px; background: rgba(155, 135, 245, 0.18); border: 1px solid #9B87F5;
        color: #DDD6FE; font-size: 0.85rem; font-family: ui-monospace, monospace;
      }
      .chip.dim { background: transparent; border-color: #444444; color: #9CA3AF; }
      .stats { display: flex; gap: 0.75rem; margin-top: 0.75rem; }
      .stat {
        flex: 1; background: #1F2023; border: 1px solid #444444; border-radius: 20px;
        padding: 0.75rem 1.1rem; color: #F3F4F6; font-size: 1.15rem; font-variant-numeric: tabular-nums;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Sidebar: which API to call ---------------------------------------------

with st.sidebar:
    st.subheader("Connection")
    api_url = st.text_input("API URL", value=API_URL, help="Your local API or your Render URL.").strip().rstrip("/")
    # Never pre-filled: widget values are sent to the browser, and the server's APP_TOKEN must stay on the server.
    token_input = st.text_input(
        "App token", type="password",
        help="Leave empty to use APP_TOKEN from the UI server's environment.",
    )
    token = token_input or APP_TOKEN
    st.caption("Using the token you entered." if token_input else
               "Using APP_TOKEN from the environment." if APP_TOKEN else "No token set.")
    check = st.button("Check connection", use_container_width=True)


def server_is_awake(url: str) -> bool:
    try:
        return requests.get(f"{url}/", timeout=2).ok
    except requests.RequestException:
        return False


def wake_server(url: str) -> None:
    """Any request wakes a sleeping Render instance; this one just waits for it to come up."""
    try:
        requests.get(f"{url}/", timeout=90)
    except requests.RequestException:
        pass


# Render's free plan sleeps after 15 idle minutes. Start waking it as soon as the page
# opens, so it's warming up while the user types.
awake = server_is_awake(api_url)
if awake:
    st.session_state.wake_started = False
elif st.session_state.get("wake_started") != api_url:
    st.session_state.wake_started = api_url
    threading.Thread(target=wake_server, args=(api_url,), daemon=True).start()


def as_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def answer_card(body_html: str, label: str = "answer", refused: bool = False) -> str:
    css = "answer-card refused" if refused else "answer-card"
    return f'<div class="{css}"><div class="label">{label}</div><div>{body_html}</div></div>'


def result_card(data: dict) -> str:
    """The final answer, with its citations, or a clearly marked refusal."""
    if not data["citations"]:
        return answer_card(
            as_html(data["answer"])
            + '<div class="sources"><div class="label">sources</div>'
              '<span class="muted">None. The documents don\'t cover this question, so the model refused '
              "instead of guessing.</span></div>",
            label="not in the documents",
            refused=True,
        )
    chips = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in data["citations"])
    return answer_card(
        as_html(data["answer"]) + f'<div class="sources"><div class="label">cited documents</div>{chips}</div>'
    )


def stats_row(data: dict) -> str:
    stats = [("tokens_used", f'{data["tokens_used"]:,}'), ("cost_usd", f'${data["cost_usd"]:.6f}')]
    if "chunks_indexed" in data:
        stats.insert(0, ("chunks_indexed", data["chunks_indexed"]))
    cells = "".join(f'<div class="stat"><div class="label">{k}</div>{v}</div>' for k, v in stats)
    return f'<div class="stats">{cells}</div>'


def retrieved_chunks(data: dict) -> None:
    """Every chunk the API retrieved for the question, with the cited ones highlighted."""
    chunk_ids = data.get("retrieved_chunk_ids", [])
    with st.expander(f"Retrieved chunks ({len(chunk_ids)})"):
        if not chunk_ids:
            st.caption("Nothing was retrieved. Is the index empty?")
            return
        chips = "".join(
            f'<span class="chip{"" if c.split("#")[0] in data["citations"] else " dim"}">{html.escape(c)}</span>'
            for c in chunk_ids
        )
        st.markdown(chips, unsafe_allow_html=True)
        st.caption("Best match first. Highlighted chunks belong to a cited document.")


def error_detail(resp: requests.Response):
    try:
        body = resp.json()
    except ValueError:
        return resp.text
    detail = body.get("detail") or body.get("error") or body
    if isinstance(detail, list):  # FastAPI validation errors (422)
        return "; ".join(f'{".".join(str(p) for p in e.get("loc", [])[1:])}: {e.get("msg")}' for e in detail)
    return detail


def sse_events(resp: requests.Response):
    """Yield (event, data) pairs from a server-sent-events response as they arrive."""
    event = None
    for line in resp.iter_lines(chunk_size=None, decode_unicode=True):
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            yield event, json.loads(line[len("data: "):])


if check:
    with st.sidebar:
        if not token:
            st.error("Enter an app token first.")
        else:
            try:
                r = requests.get(f"{api_url}/debug/pinecone", headers={"X-App-Token": token}, timeout=(10, 90))
                if r.ok:
                    info = r.json()
                    st.success(f"Connected. Index `{info['index']}` holds {info['vector_count']} chunks.")
                else:
                    st.error(f"API returned {r.status_code}: {error_detail(r)}")
            except requests.RequestException:
                st.error(f"Could not reach {api_url}.")


ask_tab, ingest_tab = st.tabs(["Ask", "Add document"])

# --- Ask: POST /ask/stream ----------------------------------------------------

with ask_tab:
    with st.form("ask_form", clear_on_submit=False, border=False):
        question = st.text_area(
            "Question",
            placeholder="Ask about your documents, e.g. How many remote days are allowed?",
            label_visibility="collapsed",
            height=68,
            max_chars=4000,
        )
        left, right = st.columns([10, 1], vertical_alignment="center")
        force_bad = ALLOW_FORCE_BAD and left.toggle(
            "Simulate bad output",
            help="Sends force_bad=true: the model is told to return malformed JSON, "
                 "so you can watch the output guardrail reject it.",
        )
        submitted = right.form_submit_button("↑", help="Send")

    if not submitted and not awake:
        st.markdown(
            '<div class="note">☕ <b>Waking up the server…</b> It sleeps after 15 minutes without use '
            "(free hosting), so your first answer may take up to a minute. You can type your question now.</div>",
            unsafe_allow_html=True,
        )

    if submitted:
        if not question.strip():
            st.warning("Type a question first.")
        elif not token:
            st.error("No app token. Enter one in the sidebar or set APP_TOKEN in .env.")
        else:
            output = st.empty()
            waiting = "Searching your documents…" if awake else "☕ Waking up the server, this can take up to a minute…"
            output.markdown(answer_card(f'<span class="muted">{waiting}</span>'), unsafe_allow_html=True)
            finished = False
            try:
                with requests.post(
                    f"{api_url}/ask/stream",
                    json={"question": question, "force_bad": force_bad},
                    headers={"X-App-Token": token},
                    stream=True,
                    timeout=(10, 120),  # (connect, wait between chunks): covers a cold start
                ) as resp:
                    if not resp.ok:
                        output.empty()
                        st.error(f"API returned {resp.status_code}: {error_detail(resp)}")
                        st.stop()

                    resp.encoding = "utf-8"
                    text = ""
                    for event, data in sse_events(resp):
                        if event == "delta":
                            text += data["text"]
                            output.markdown(answer_card(as_html(text) + '<span class="cursor">▍</span>'),
                                            unsafe_allow_html=True)
                        elif event == "done":
                            output.markdown(result_card(data) + stats_row(data), unsafe_allow_html=True)
                            retrieved_chunks(data)
                            finished = True
                        elif event == "error":
                            # The guardrail rejected the full reply: remove any text already shown.
                            output.empty()
                            st.error(f"API returned {data['status']}: {data['detail']}")
                            finished = True
            except requests.RequestException:
                output.empty()
                st.error(f"Could not reach the API at {api_url}. Is it running?")
                st.stop()

            if not finished:
                output.empty()
                st.error("The answer was cut off before it finished. Please try again.")

# --- Add document: POST /ingest ----------------------------------------------

with ingest_tab:
    with st.container(key="ingest_panel"):
        document_id = st.text_input("document_id", placeholder="e.g. POL-101",
                                    help="Sending an existing document_id replaces that document.")
        source = st.text_input("source (optional)", placeholder="e.g. doc1_handbook.txt")
        doc_text = st.text_area("Text", height=220, placeholder="Paste the document's text here…")
        add = st.button("Add to knowledge base", type="primary")

    if add:
        if not token:
            st.error("No app token. Enter one in the sidebar or set APP_TOKEN in .env.")
        else:
            body = {"document_id": document_id, "text": doc_text}
            if source.strip():
                body["source"] = source
            try:
                with st.spinner("Chunking and embedding…"):
                    r = requests.post(f"{api_url}/ingest", json=body, headers={"X-App-Token": token},
                                      timeout=(10, 120))
            except requests.RequestException:
                st.error(f"Could not reach the API at {api_url}. Is it running?")
                st.stop()
            if r.ok:
                data = r.json()
                st.markdown(
                    answer_card(f'Indexed <span class="chip">{html.escape(data["document_id"])}</span> '
                                f'as {data["chunks_indexed"]} chunk{"" if data["chunks_indexed"] == 1 else "s"}.',
                                label=data["status"])
                    + stats_row(data),
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"API returned {r.status_code}: {error_detail(r)}")
