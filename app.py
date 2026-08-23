"""Streamlit chat interface for the e& Egypt assistant.

    streamlit run app.py

Chat bubbles are rendered as custom HTML rather than st.chat_message, which
gives control over alignment, colour and avatars. Answers arrive as markdown,
so they are converted to HTML before being placed inside a bubble - markdown
is not processed inside raw HTML blocks.

The bot avatar changes with the outcome of the request, so the face is a
visual read-out of which branch the agent took.

Three pieces of timing work sit in this file:

Loading the embedding model costs around fifteen seconds, almost all of it
importing torch and sentence-transformers rather than reading the weights.
Doing that before the first render left the page blank for the whole wait, so
it now happens on a background thread and the page paints straight away.

A question is handled across two reruns rather than one. The first rerun only
records the question and paints it; the second runs the agent. Streamlit paints
nothing until a script run finishes, so doing both in one pass left the
customer's own message invisible for the whole round-trip.

The answer is painted as it is written rather than when it is finished. The
agent runs on its own thread and the reply is read back a piece at a time.
"""

import base64
import re
import threading
import time

import markdown as md
import streamlit as st

from src.config import EMBED_MODEL, PROJECT_ROOT
from src.graph.build import AnswerStream
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

# A USSD service code: opens with *, closes with #, digits and asterisks in
# between. *130#, *319*155# and *556*9*RechargeCode# all match. Ordinary
# emphasis does not - *like this* has no closing hash.
USSD_PATTERN = re.compile(r"\*[0-9A-Za-z*]+#")

# How often the growing answer is repainted, in seconds. Every piece would
# mean a redraw per token - the browser cannot keep up and the text judders.
# At this interval the writing looks continuous and the work is negligible.
REDRAW_INTERVAL = 0.06

STYLES = """
<style>
/* e& brand palette: red #E00800 is the primary, Heath #4B0F1E the dark
   companion. The greens this replaced were the pre-2022 Etisalat identity. */
/* tighten the page and leave room for the input bar */
.block-container { max-width: 820px; padding-top: 2rem; padding-bottom: 6rem; }

.row { display: flex; margin: 14px 0; align-items: flex-end; gap: 10px; }
.row.bot  { justify-content: flex-start; }
.row.user { justify-content: flex-end; }

.avatar {
    width: 38px; height: 38px; border-radius: 50%;
    object-fit: cover; flex-shrink: 0;
    background: #4b0f1e;
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
    background: #2b1418;
    color: #fafafa;
    border-bottom-left-radius: 4px;
}
.bubble.user {
    background: #e00800;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}

/* markdown inside a bubble should not add outer spacing */
.bubble p:first-child { margin-top: 0; }
.bubble p:last-child  { margin-bottom: 0; }
.bubble ul { margin: 6px 0; padding-left: 20px; }
.bubble strong { font-weight: 600; }

/* the block caret shown at the end of an answer still being written */
.caret {
    display: inline-block;
    width: 7px; height: 15px;
    margin-left: 2px;
    vertical-align: -2px;
    background: #e00800;
    animation: caret 1s steps(2, start) infinite;
}

@keyframes caret { 50% { opacity: 0; } }

/* service codes: monospace, and forced left-to-right so *319*155# keeps its
   asterisks and hash in the right places inside an Arabic sentence */
.bubble code {
    font-family: "SF Mono", "Cascadia Mono", Consolas, monospace;
    font-size: 0.9em;
    padding: 1px 6px;
    border-radius: 5px;
    background: rgba(255, 255, 255, 0.10);
    white-space: nowrap;
    direction: ltr;
    unicode-bidi: isolate;
}
.bubble.user code { background: rgba(0, 0, 0, 0.18); }

/* sources sit under the bot bubble, indented past the avatar */
.sources-wrap { margin: -6px 0 16px 48px; }

/* the placeholder shown while the agent is working */
.bubble.typing { padding: 14px 16px; }
.typing-dots { display: flex; gap: 5px; align-items: center; }
.typing-dots span {
    width: 7px; height: 7px; border-radius: 50%;
    background: #9ca3af;
    animation: blink 1.3s infinite ease-in-out both;
}
.typing-dots span:nth-child(2) { animation-delay: 0.18s; }
.typing-dots span:nth-child(3) { animation-delay: 0.36s; }

@keyframes blink {
    0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
    40%           { opacity: 1;    transform: translateY(-2px); }
}

/* the panel shown while the embedding model loads in the background */
.warming-note {
    display: flex; align-items: center; gap: 10px;
    margin: 4px 0 18px 0;
    color: #9ca3af; font-size: 0.9rem;
}
.warming-note .spinner {
    width: 13px; height: 13px; flex-shrink: 0;
    border: 2px solid #4b0f1e;
    border-top-color: #e00800;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.skeleton {
    height: 42px; border-radius: 18px;
    background: linear-gradient(
        90deg, #2b1418 25%, #3d1d23 50%, #2b1418 75%
    );
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite linear;
}
.skeleton.short { width: 46%; }
.skeleton.mid   { width: 62%; }
.skeleton.long  { width: 54%; }

@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
</style>
"""

WARMING_MESSAGE = "Loading the knowledge base · جارٍ تحميل قاعدة المعرفة"


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

    Service codes are pulled out before the markdown conversion and put back
    afterwards. Markdown reads the asterisks in *319*155# as emphasis, renders
    319 in italics and swallows both of them, leaving 319155# on screen - a
    code the customer cannot dial. Since a dial code is the most actionable
    thing this assistant produces, it has to survive intact.
    """
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"\[(\d+)†[^\]]*\]", r"[\1]", text)

    codes = []

    def stash(match):
        codes.append(match.group(0))
        # alphanumeric, so markdown passes it through untouched
        return f"ZZUSSD{len(codes) - 1}ZZ"

    text = USSD_PATTERN.sub(stash, text)

    html = md.markdown(text, extensions=["nl2br"])

    for index, code in enumerate(codes):
        html = html.replace(
            f"ZZUSSD{index}ZZ",
            f'<code class="ussd" dir="ltr">{code}</code>',
        )

    # a code the model had already wrapped in backticks comes back doubly
    # wrapped, which renders as a box inside a box
    return re.sub(r"<code>(<code class=\"ussd\".*?</code>)</code>", r"\1", html)


def bubble_html(role, content, mood="joyful", writing=False):
    """One chat row as markup: avatar plus bubble, aligned by who is speaking.

    Returned rather than written, because a bubble being streamed into is
    rebuilt many times inside a single placeholder.

    writing adds a blinking caret, which is what separates an answer still
    arriving from one that stopped early.
    """
    direction = "rtl" if detect_language(content) == "ar" else "ltr"
    align = "right" if direction == "rtl" else "left"
    body = to_html(content)

    if writing:
        body += '<span class="caret"></span>'

    if role == "user":
        return (
            f'<div class="row user">'
            f'<div class="bubble user" dir="{direction}" '
            f'style="text-align: {align};">{body}</div>'
            f'</div>'
        )

    return (
        f'<div class="row bot">'
        f'{avatar_html(mood)}'
        f'<div class="bubble bot" dir="{direction}" '
        f'style="text-align: {align};">{body}</div>'
        f'</div>'
    )


def render_bubble(role, content, mood="joyful"):
    """Write one finished chat row to the page."""
    st.markdown(bubble_html(role, content, mood), unsafe_allow_html=True)


def typing_html():
    """The bot bubble shown between sending a question and the first word of
    the answer - retrieval and any rewriting happen in that gap."""
    return (
        f'<div class="row bot">'
        f'{avatar_html("joyful")}'
        f'<div class="bubble bot typing">'
        f'<div class="typing-dots"><span></span><span></span><span></span></div>'
        f'</div>'
        f'</div>'
    )


def render_warming():
    """What fills the conversation area while the model loads.

    Skeleton bubbles rather than a bare spinner: they show where the
    conversation is about to appear, so the shape of the page does not jump
    when the real content arrives.
    """
    rows = "".join(
        f'<div class="row {side}">'
        f'<div class="skeleton {width}"></div>'
        f'</div>'
        for side, width in (("bot", "mid"), ("user", "short"), ("bot", "long"))
    )

    st.markdown(
        f'<div class="warming-note">'
        f'<div class="spinner"></div>{WARMING_MESSAGE}</div>{rows}',
        unsafe_allow_html=True,
    )


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
def warm_up():
    """Start loading the collection on a background thread.

    Returns immediately with a handle describing the load. Opening the
    collection constructs the embedding function, which imports torch and
    sentence-transformers - about fifteen seconds, and the reason the page used
    to sit blank. Nothing above the chat area needs the collection, so the page
    can render while this runs.

    Cached, so the thread is started once for the whole server rather than once
    per browser session.
    """
    handle = {"collection": None, "error": None}

    # set up on this thread: the log file should exist even if the load fails
    setup_logging("app.log")

    def load():
        try:
            handle["collection"] = get_collection()
        except Exception as error:            # noqa: BLE001 - reported in the UI
            handle["error"] = error

    thread = threading.Thread(target=load, name="warm-up", daemon=True)
    thread.start()

    handle["thread"] = thread
    return handle


def wait_for_collection(handle):
    """Block until the background load has finished.

    Every path that needs the collection comes through here, so the loading
    only ever happens on the warm-up thread. Letting the main thread call
    get_collection() while the warm-up thread is inside it would build the
    embedding model twice at once.
    """
    handle["thread"].join()

    if handle["error"] is not None:
        raise RuntimeError(f"the knowledge base failed to load: "
                           f"{handle['error']}")

    return handle["collection"]


def submit_question(question):
    """Record a question and hand control straight back to Streamlit.

    Half of the exchange: this pass ends immediately after the rerun, so the
    customer's message is painted before the agent is asked anything.
    """
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.pending = question
    st.rerun()


def stream_answer(question, history, placeholder):
    """Run the agent, painting the reply into placeholder as it is written.

    Returns the finished answer and its sources. Raises RuntimeError for
    anything the agent could not recover from, which the caller turns into a
    message in the chat rather than a stack trace on the page.
    """
    stream = AnswerStream(question, history)

    text = ""
    painted = 0.0

    for piece in stream.tokens():
        text += piece
        now = time.monotonic()

        # repaint on a timer rather than per piece, so a fast provider does
        # not queue up hundreds of redraws the browser has to work through
        if now - painted >= REDRAW_INTERVAL:
            painted = now
            placeholder.markdown(
                bubble_html("assistant", text, writing=True),
                unsafe_allow_html=True,
            )

    result = stream.result or {}

    # the state's answer rather than the accumulated text: they agree, but the
    # state is what every other caller of the agent sees
    answer = result.get("answer") or text

    return answer, result.get("sources", [])


def resolve_pending(handle):
    """Run the agent for the question recorded by the previous rerun.

    The question is already the last entry in messages by this point, so the
    history handed to the agent is everything before it.
    """
    question = st.session_state.pending

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[:-1]
    ]

    # one slot, reused: dots first, then the answer growing in their place
    placeholder = st.empty()
    placeholder.markdown(typing_html(), unsafe_allow_html=True)

    try:
        # a question asked during warm-up waits here, showing the typing dots
        wait_for_collection(handle)

        answer, sources = stream_answer(question, history, placeholder)
        mood = pick_mood(answer, sources)

    except RuntimeError as error:
        answer = f"Sorry, something went wrong: {error}"
        sources = []
        mood = "sad"

    # the caret goes away, and the rerun below redraws this as a normal message
    placeholder.markdown(
        bubble_html("assistant", answer, mood),
        unsafe_allow_html=True,
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "mood": mood,
    })

    # cleared before the rerun, so a failed answer cannot loop forever
    st.session_state.pending = None
    st.rerun()


def render_sidebar(handle):
    """The About panel. The chunk count is the one thing here that needs the
    collection, so it shows a placeholder until the load finishes rather than
    holding up the rest of the page."""
    with st.sidebar:
        st.header("About")
        st.write(
            "Ask about e& Egypt mobile packages, internet plans, prices and "
            "service codes. Answers come from the official documentation only."
        )

        st.divider()

        collection = handle["collection"]

        if handle["error"] is not None:
            st.caption("**Knowledge base:** failed to load")
        elif collection is None:
            st.caption("**Knowledge base:** loading…")
        else:
            st.caption(f"**Knowledge base:** {collection.count()} chunks")

        st.caption(f"**Embeddings:** {EMBED_MODEL}")
        st.caption(f"**Model:** {MODEL}")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending = None
            st.rerun()


def main():
    st.set_page_config(page_title="e& Egypt Assistant", page_icon="💬")
    st.markdown(STYLES, unsafe_allow_html=True)

    # returns at once; the model loads on a thread behind this call
    handle = warm_up()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending" not in st.session_state:
        st.session_state.pending = None

    render_sidebar(handle)

    st.title("e& Egypt Assistant")
    st.caption("اسأل بالعربية أو بالإنجليزية · Ask in Arabic or English")

    ready = handle["collection"] is not None
    warming = not ready and handle["error"] is None

    if handle["error"] is not None:
        st.error(f"The knowledge base could not be loaded: {handle['error']}")

    if not st.session_state.messages and not st.session_state.pending:
        st.write("")
        columns = st.columns(2)

        for position, suggestion in enumerate(SUGGESTIONS):
            with columns[position % 2]:
                # left enabled while warming: a question asked now simply waits
                if st.button(suggestion, key=f"s{position}",
                             use_container_width=True):
                    submit_question(suggestion)

    for message in st.session_state.messages:
        render_bubble(
            message["role"],
            message["content"],
            message.get("mood", "joyful"),
        )

        if message.get("sources") and message.get("mood") != "surprised":
            render_sources(message["sources"])

    question = st.chat_input(
        "Ask a question..." if ready else "Ask away - still warming up..."
    )

    if question:
        submit_question(question)

    # everything above has been painted by now, so the customer sees their own
    # message and the answer arriving under it
    if st.session_state.pending:
        resolve_pending(handle)

    # nothing to answer, but the model is still loading: show the skeleton,
    # wait for it, then redraw the page with the real chunk count in place
    elif warming:
        render_warming()
        wait_for_collection(handle)
        st.rerun()


if __name__ == "__main__":
    main()
