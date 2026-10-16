import html
import json
import os
import threading

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# The UI calls the API from the server side, so APP_TOKEN never reaches the browser.
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
      .block-container { max-width: 760px; padding-top: 32vh; }

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

      /* Results */
      .answer-card {
        background: #1F2023; border: 1px solid #444444; border-radius: 24px;
        padding: 1.25rem 1.5rem; margin-top: 1rem; color: #E5E7EB; line-height: 1.6;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
      }
      .muted { color: #9CA3AF; }
      .cursor { color: #9B87F5; animation: blink 1s steps(1) infinite; }
      @keyframes blink { 50% { opacity: 0; } }
      .note {
        margin-top: 0.9rem; padding: 0.7rem 1.1rem; border-radius: 16px;
        background: rgba(31, 32, 35, 0.85); border: 1px solid #444444;
        color: #D1D5DB; font-size: 0.9rem; line-height: 1.5;
      }
      .label { color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
      .stats { display: flex; gap: 0.75rem; margin-top: 0.75rem; }
      .stat {
        flex: 1; background: #1F2023; border: 1px solid #444444; border-radius: 20px;
        padding: 0.75rem 1.1rem; color: #F3F4F6; font-size: 1.15rem; font-variant-numeric: tabular-nums;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.form("ask_form", clear_on_submit=False, border=False):
    question = st.text_area(
        "Question",
        placeholder="Type your message here...",
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


def server_is_awake() -> bool:
    try:
        return requests.get(f"{API_URL}/", timeout=2).ok
    except requests.RequestException:
        return False


def wake_server() -> None:
    """Any request wakes a sleeping Render instance; this one just waits for it to come up."""
    try:
        requests.get(f"{API_URL}/", timeout=90)
    except requests.RequestException:
        pass


# Render's free plan sleeps after 15 idle minutes. Start waking it as soon as the page
# opens, so it's warming up while the user types.
awake = server_is_awake()
if awake:
    st.session_state.wake_started = False
elif not st.session_state.get("wake_started"):
    st.session_state.wake_started = True
    threading.Thread(target=wake_server, daemon=True).start()


def as_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def answer_card(body_html: str) -> str:
    return f'<div class="answer-card"><div class="label">answer</div><div>{body_html}</div></div>'


def stats_row(data: dict) -> str:
    return (
        '<div class="stats">'
        f'<div class="stat"><div class="label">tokens_used</div>{data["tokens_used"]:,}</div>'
        f'<div class="stat"><div class="label">cost_usd</div>${data["cost_usd"]:.6f}</div>'
        "</div>"
    )


def error_detail(resp: requests.Response):
    try:
        body = resp.json()
        return body.get("detail") or body.get("error") or body
    except ValueError:
        return resp.text


def sse_events(resp: requests.Response):
    """Yield (event, data) pairs from a server-sent-events response as they arrive."""
    event = None
    for line in resp.iter_lines(chunk_size=None, decode_unicode=True):
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            yield event, json.loads(line[len("data: "):])


if not submitted and not awake:
    st.markdown(
        '<div class="note">☕ <b>Waking up the server…</b> It sleeps after 15 minutes without use '
        "(free hosting), so your first answer may take up to a minute. You can type your question now.</div>",
        unsafe_allow_html=True,
    )

if submitted:
    if not question.strip():
        st.warning("Type a question first.")
    elif not APP_TOKEN:
        st.error("APP_TOKEN is not set in .env.")
    else:
        output = st.empty()
        waiting = "Thinking…" if awake else "☕ Waking up the server, this can take up to a minute…"
        output.markdown(answer_card(f'<span class="muted">{waiting}</span>'), unsafe_allow_html=True)
        finished = False
        try:
            with requests.post(
                f"{API_URL}/ask/stream",
                json={"question": question, "force_bad": force_bad},
                headers={"X-App-Token": APP_TOKEN},
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
                        output.markdown(answer_card(as_html(data["answer"])) + stats_row(data),
                                        unsafe_allow_html=True)
                        finished = True
                    elif event == "error":
                        # The guardrail rejected the full reply: remove any text already shown.
                        output.empty()
                        st.error(f"API returned {data['status']}: {data['detail']}")
                        finished = True
        except requests.RequestException:
            output.empty()
            st.error(f"Could not reach the API at {API_URL}. Is it running?")
            st.stop()

        if not finished:
            output.empty()
            st.error("The answer was cut off before it finished. Please try again.")
