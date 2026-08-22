"""Streamlit chat interface for the e& Egypt assistant.

    streamlit run app.py

Chat bubbles are rendered as custom HTML rather than st.chat_message, which
gives control over alignment, colour and avatars. Answers arrive as markdown,
so they are converted to HTML before being placed inside a bubble - markdown
is not processed inside raw HTML blocks.

The bot avatar changes with the outcome of the request, so the face is a
visual read-out of which branch the agent took.
"""

import base64
import re 

import markdown as md
import streamlit as st

from src.config import EMBED_MODEL, PROJECT_ROOT
from src.graph.build import ask
from src.language import detect_language
from src.llm import MODEL
from src.logging_config import setup_logging
from src.vectorstore import get_collection

ASSETS_DIR = PROJECT_ROOT / "assets"

AVATARS = {
    "welcoming": "welcoming.png",   # greeting, no retrieval
    "joyful": "joyful.png",         # answered from the knowledge base
    "surprised": "surprised.png",   # off topic or declined
    "sad": "sad.png",               # something went wrong
}

SUGGESTIONS = [
    "How much does Emerald 430 cost per month?",
    "Can I use a Data Line SIM in my phone?",
    "كام مكس يساوي دقيقة لشبكة تانية؟",
    "ايه الكود بتاع أقوى كارت؟",
]

STYLES = """
<style>
/* tighten the page and leave room for the input bar */
.block-container { max-width: 820px; padding-top: 2rem; padding-bottom: 6rem; }

.row { display: flex; margin: 14px 0; align-items: flex-end; gap: 10px; }
.row.bot  { justify-content: flex-start; }
.row.user { justify-content: flex-end; }

.avatar {
    width: 38px; height: 38px; border-radius: 50%;
    object-fit: cover; flex-shrink: 0;
    background: #1f2937;
}
.avatar.fallback {
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}

.bubble {
    max-width: 78%;
    padding: 12px 16px;
    border-radius: 18px;
    line-height: 1.55;
    font-size: 0.95rem;
    word-wrap: break-word;
}
.bubble.bot {
    background: #262730;
    color: #fafafa;
    border-bottom-left-radius: 4px;
}
.bubble.user {
    background: #00a862;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}

/* markdown inside a bubble should not add outer spacing */
.bubble p:first-child { margin-top: 0; }
.bubble p:last-child  { margin-bottom: 0; }
.bubble ul { margin: 6px 0; padding-left: 20px; }
.bubble strong { font-weight: 600; }

/* sources sit under the bot bubble, indented past the avatar */
.sources-wrap { margin: -6px 0 16px 48px; }
</style>
"""


@st.cache_data
def encode_avatar(filename):
    """Base64-encode an avatar once and reuse it. Cached because the same
    image would otherwise be re-encoded on every rerun."""
    path = ASSETS_DIR / filename

    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def avatar_html(mood="joyful"):
    """The bot avatar for a given mood, as an inline image.

    The image is base64-encoded so it can live inside the same HTML block as
    the bubble - Streamlit cannot mix st.image into custom markup.
    """
    encoded = encode_avatar(AVATARS.get(mood, "joyful.png"))

    if encoded is None:
        return '<div class="avatar fallback">💬</div>'

    return f'<img class="avatar" src="data:image/png;base64,{encoded}">'


def to_html(text):
    """Markdown to HTML, so formatting survives inside a bubble.

    The model emits citations in inconsistent formats - full-width CJK
    brackets, and sometimes a line-range suffix like [1†L1-L4] - despite the
    prompt asking for plain [1]. Normalising here is more reliable than
    hoping the instruction holds.
    """
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"\[(\d+)†[^\]]*\]", r"[\1]", text)
    return md.markdown(text, extensions=["nl2br"])


def render_bubble(role, content, mood="joyful"):
    """One chat row: avatar plus bubble, aligned by who is speaking."""
    direction = "rtl" if detect_language(content) == "ar" else "ltr"
    align = "right" if direction == "rtl" else "left"
    body = to_html(content)

    if role == "user":
        html = (
            f'<div class="row user">'
            f'<div class="bubble user" dir="{direction}" '
            f'style="text-align: {align};">{body}</div>'
            f'</div>'
        )
    else:
        html = (
            f'<div class="row bot">'
            f'{avatar_html(mood)}'
            f'<div class="bubble bot" dir="{direction}" '
            f'style="text-align: {align};">{body}</div>'
            f'</div>'
        )

    st.markdown(html, unsafe_allow_html=True)


def render_sources(sources):
    """Collapsible source list under a bot message."""
    st.markdown('<div class="sources-wrap">', unsafe_allow_html=True)

    with st.expander(f"Sources ({len(sources)})"):
        for number, source in enumerate(sources, start=1):
            st.caption(
                f"**[{number}]** {source['doc']} — {source['section']}  \n"
                f"{source['lang']} · distance {source['distance']}"
            )

    st.markdown("</div>", unsafe_allow_html=True)


def pick_mood(answer, sources):
    """Choose an avatar from what the agent did.

    Detecting a refusal by looking for apology words is crude and will
    occasionally misfire. A cleaner version would have the generate node
    return an explicit flag in the state.
    """
    if not sources:
        return "welcoming"          # smalltalk: retrieval never ran

    apologies = ("sorry", "i can't", "i cannot", "عذر", "آسف", "لا يمكنني")
    lowered = answer.lower()

    if any(word in lowered for word in apologies):
        return "surprised"

    return "joyful"


@st.cache_resource
def start_up():
    """Load the collection once. Streamlit reruns this file on every
    interaction, so without caching the embedding model would reload each
    time and the app would be unusable."""
    setup_logging("app.log")
    return get_collection()


def handle_question(question):
    """Run the agent and store both sides of the exchange."""
    # the history as it stands before this question is added
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": question})

    try:
        result = ask(question, history)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        mood = pick_mood(answer, sources)

    except RuntimeError as error:
        answer = f"Sorry, something went wrong: {error}"
        sources = []
        mood = "sad"

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "mood": mood,
    })


def main():
    st.set_page_config(page_title="e& Egypt Assistant", page_icon="💬")
    st.markdown(STYLES, unsafe_allow_html=True)

    collection = start_up()

    with st.sidebar:
        st.header("About")
        st.write(
            "Ask about e& Egypt mobile packages, internet plans, prices and "
            "service codes. Answers come from the official documentation only."
        )

        st.divider()
        st.caption(f"**Knowledge base:** {collection.count()} chunks")
        st.caption(f"**Embeddings:** {EMBED_MODEL}")
        st.caption(f"**Model:** {MODEL}")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("e& Egypt Assistant")
    st.caption("اسأل بالعربية أو بالإنجليزية · Ask in Arabic or English")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.write("")
        columns = st.columns(2)

        for position, suggestion in enumerate(SUGGESTIONS):
            with columns[position % 2]:
                if st.button(suggestion, key=f"s{position}",
                             use_container_width=True):
                    handle_question(suggestion)
                    st.rerun()

    for message in st.session_state.messages:
        render_bubble(
            message["role"],
            message["content"],
            message.get("mood", "joyful"),
        )

        if message.get("sources") and message.get("mood") != "surprised":
            render_sources(message["sources"])

    question = st.chat_input("Ask a question...")

    if question:
        handle_question(question)
        st.rerun()


if __name__ == "__main__":
    main()