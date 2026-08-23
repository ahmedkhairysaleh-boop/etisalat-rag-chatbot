"""The agent's nodes.

Each node takes the state, does one job, and returns only the fields it
changed. LangGraph merges the result into the state before the next node runs.

Any node that produces the customer-facing reply goes through _reply(), which
streams the text out through the state's sink when an interface is listening
and falls back to an ordinary blocking call when one is not.
"""

import logging

from src.language import detect_language
from src.llm import generate as call_llm
from src.llm import generate_stream as call_llm_stream
from src.prompts import (SYSTEM_PROMPT, NO_CONTEXT_PROMPT, REWRITE_PROMPT,
                         DECOMPOSE_PROMPT, build_user_prompt,
                         build_rewrite_prompt, parse_queries)
from src.vectorstore import get_collection, search

log = logging.getLogger(__name__)

# A floor, not a topic filter.
#
# Measured across the 24 test questions: answerable questions retrieve at
# 0.094-0.183, off-topic ones at 0.170-0.228. The ranges overlap, so no
# distance threshold can separate them - every chunk is telecom text in the
# same two languages, and even a weather question finds something loosely
# similar. Vector distance measures textual similarity, not whether a chunk
# answers the question.
#
# So this only catches catastrophic retrieval failure. Off-topic questions are
# handled by the system prompt instead, which refuses them correctly.
RELEVANCE_LIMIT = 0.30

# Questions longer than this usually stand on their own, so rewriting them
# would spend an API call for nothing. Follow-ups tend to be short:
# "ايه تفاصيلها", "and the price?", "what about 780?"
REWRITE_LENGTH_LIMIT = 60

# The product families the knowledge base describes, written the ways a
# customer writes them - both scripts, and the spellings that differ only in
# their hamza.
#
# A question naming one of these says what it is about, so there is nothing for
# the rewriter to fill in. Length alone used to decide this, and it was wrong:
# "Can I add my landline to my Emerald account?" is 44 characters and complete,
# but the rewriter, asked to supply what the customer left implicit, would
# reach into the previous turns and pull out a tier the customer never
# mentioned - turning it into a question about Emerald 1120. Retrieval then
# narrowed to that tier, and a question about a whole product family was
# answered as though it were about one package.
PRODUCT_TERMS = (
    "emerald", "إيميرالد", "ايميرالد", "إميرالد", "اميرالد",
    "hekaya", "حكاية", "حكايه",
    "mixat", "ميكسات",
    # "Mixes" is how a customer refers to the Hekaya Mixat unit, and it is
    # every bit as anchoring as the bundle name. Leaving it out sent "If I
    # have rolled-over Mixes, which get used first?" to the rewriter, which
    # returned it as a question about Emerald 430.
    "mix", "mixes", "ميكس",
    "data line", "dataline", "داتا لاين", "خط الداتا", "خط داتا",
    "aqwa", "أقوى كارت", "اقوى كارت",
    "ahlan", "أهلاً", "أهلا", "اهلا",
    "menu", "منيو",
    "super connect", "سوبر كونكت",
    "coins", "كوينز",
)

# How many past turns the rewriter sees. More context rarely helps and every
# token counts against the rate limit.
HISTORY_TURNS = 4

# Multi-product questions are usually long enough to name both products, so
# short questions are not worth a decomposition call.
DECOMPOSE_LENGTH_MIN = 25

# Cap on merged results.
#
# Lowered from 8 to 5. The free tier allows 8,000 tokens a minute, and the
# retrieved context is most of what a question spends, so eight chunks put a
# second question in the same minute out of reach. Five keeps enough material
# for a comparison while leaving room to ask a follow-up.
MERGED_HIT_LIMIT = 5

# Words that mark a question as comparing two things.
#
# Decomposition only helps when a question asks about two products at once -
# that is the whole case the prompt describes. Without a marker it was firing
# on nearly every question over 25 characters and spending an API call to be
# told "no split needed". On a tokens-per-minute budget that call is not free:
# it is a third of the requests and a slice of the tokens, for nothing.
COMPARISON_MARKERS = (
    " vs ", "vs.", "versus", "compare", "comparison", "difference", "differ",
    "better", "cheaper", "costs more", "more expensive", "which costs",
    "which gives", "which is", "which one",
    "الفرق", "مقارنة", "قارن", "أفضل", "افضل", "أرخص", "ارخص",
    "أغلى", "اغلى", "أيهما", "ايهما", " ولا ",
)

GREETINGS = {
    "hi", "hello", "hey", "good morning", "good evening",
    "السلام عليكم", "سلام", "اهلا", "أهلا", "مرحبا", "صباح الخير",
    "مساء الخير", "ازيك", "إزيك",
}

CLOSINGS = {
    "thanks", "thank you", "ok thanks", "okay thanks", "ty", "cheers",
    "bye", "goodbye", "ok", "okay", "great", "perfect", "got it",
    "شكرا", "شكراً", "متشكر", "تمام", "ماشي", "مع السلامة", "الله يخليك",
    "تسلم", "شكرا جزيلا",
}


def _reply(state, system_prompt, user_prompt):
    """Produce the customer-facing reply, streaming it if anyone is listening.

    The sink is whatever the interface put in the state. Streamlit pushes each
    piece into a queue and paints it; the scripts and the tests set no sink at
    all and get the blocking call, unchanged.

    Either way this returns the finished answer, so every node downstream and
    every caller sees the same thing regardless of how it was produced.
    """
    sink = state.get("stream")

    if sink is None:
        return call_llm(system_prompt, user_prompt)

    pieces = []

    for piece in call_llm_stream(system_prompt, user_prompt):
        pieces.append(piece)
        sink(piece)

    return "".join(pieces)


def classify_smalltalk(text):
    """Return 'greeting', 'closing', or None.

    The length check stops 'thanks, but how much is Emerald 430?' being
    treated as small talk when it carries a real question.
    """
    cleaned = text.strip().lower().rstrip("!?.،,")

    if len(cleaned) > 20:
        return None
    if cleaned in GREETINGS:
        return "greeting"
    if cleaned in CLOSINGS:
        return "closing"
    return None


def mentions_product(text):
    """Whether the question names a product family on its own."""
    lowered = text.lower()

    return any(term in lowered for term in PRODUCT_TERMS)


def needs_decompose(question):
    """Whether this question is worth an extra call to be split up.

    Only comparisons benefit. A question about one product returns the same
    single query it went in as, so asking costs a request and a few hundred
    tokens to learn nothing - and on the free tier those tokens come out of the
    same minute the customer is waiting in.

    A comparison phrased without any of the marker words will be missed and
    searched as one query. That is the trade: the occasional weaker answer on
    an unusual phrasing, against a third of the API calls on every ordinary
    question.
    """
    if len(question.strip()) < DECOMPOSE_LENGTH_MIN:
        return False

    lowered = question.lower()

    return any(marker in lowered for marker in COMPARISON_MARKERS)


def needs_rewrite(question, history):
    """Whether this question depends on the conversation around it.

    Three conditions, and all must hold. There has to be a conversation to
    depend on. The question has to be short, because a long one has usually
    said everything it needs to. And it must not name a product family - that
    is the condition that matters, because a question naming one is anchored
    already and rewriting it can only drag in a tier from an earlier turn that
    the customer never asked about.

    What remains are the genuine follow-ups: "وبالعرض المضاعف؟", "and the
    price?", "what about 780?" - short, and naming nothing.
    """
    if not history:
        return False

    if len(question.strip()) > REWRITE_LENGTH_LIMIT:
        return False

    return not mentions_product(question)


def understand(state):
    """Prepare the search queries.

    Two steps, each a separate model call so they can fail independently:

    1. Rewriting - a short follow-up like "what are its details" carries no
       product name, so it is rewritten using recent turns.
    2. Decomposition - a question naming two products needs one search per
       product. A single search returns chunks for whichever product matches
       more of the wording and misses the other entirely.

    Both are skipped when they cannot help, to save API calls. Neither is
    streamed: the customer never sees this text, and pushing it to the
    interface would print the agent's working out into the chat.
    """
    question = state["question"]
    history = state.get("history", [])
    language = detect_language(question)

    query = question

    if needs_rewrite(question, history):
        recent = history[-HISTORY_TURNS:]
        rewritten = call_llm(REWRITE_PROMPT,
                             build_rewrite_prompt(question, recent)).strip()

        # a rewriter that returns nothing, or an essay, has misbehaved -
        # fall back to the original rather than searching on garbage
        if rewritten and len(rewritten) < 300:
            query = rewritten
            log.info("understand: rewrote %r -> %r", question, query)

    elif history:
        log.info("understand: %r stands on its own, not rewriting",
                 question[:60])

    queries = [query]

    if needs_decompose(query):
        reply = call_llm(DECOMPOSE_PROMPT, f"Question: {query}")
        queries = parse_queries(reply, query)

        if len(queries) > 1:
            log.info("understand: split into %d searches: %s",
                     len(queries), queries)

    log.info("understand: language=%s, searching for %s", language, queries)

    # query is the question with the implicit parts filled in. Retrieval has
    # always used it; generate needs it too, or it answers a fragment.
    return {
        "language": language,
        "search_queries": queries,
        "resolved_question": query,
    }


def route_after_understand(state):
    """Small talk skips retrieval - searching a package database for 'thanks'
    wastes a query and returns irrelevant chunks."""
    return "smalltalk" if classify_smalltalk(state["question"]) else "retrieve"


def smalltalk(state):
    """Reply to a greeting or a thank-you without touching the knowledge base
    or the language model."""
    kind = classify_smalltalk(state["question"])
    arabic = state.get("language", detect_language(state["question"])) == "ar"

    log.info("smalltalk: %s, %r", kind, state["question"][:40])

    if kind == "greeting":
        answer = ("أهلاً بك! أنا مساعد إي آند مصر. اسألني عن الباقات، "
                  "الأسعار، أو الأكواد.") if arabic else (
                  "Hello! I am the e& Egypt assistant. Ask me about packages, "
                  "prices, or service codes.")
    else:
        answer = ("العفو! لو عندك أي سؤال تاني عن خدمات إي آند مصر، "
                  "أنا هنا.") if arabic else (
                  "You're welcome. If you have another question about e& Egypt "
                  "services, just ask.")

    # no model call to stream from, but the interface is waiting on the sink -
    # push the whole reply so it has one way of receiving an answer
    sink = state.get("stream")

    if sink is not None:
        sink(answer)

    return {"answer": answer, "sources": []}


def retrieve(state):
    """Search on every prepared query and merge the results.

    Duplicates are dropped by chunk text: two searches often return the same
    chunk, and sending it twice wastes context. The merged list is capped so a
    three-way split does not triple the prompt size and hit the rate limit.
    """
    collection = get_collection()
    queries = state.get("search_queries") or [state["question"]]

    merged = []
    seen = set()

    for query in queries:
        for hit in search(collection, query):
            if hit["text"] in seen:
                continue
            seen.add(hit["text"])
            merged.append(hit)

    merged.sort(key=lambda hit: hit["distance"])
    merged = merged[:MERGED_HIT_LIMIT]

    usable = bool(merged) and merged[0]["distance"] <= RELEVANCE_LIMIT

    log.info("retrieve: %d queries -> %d hits, best distance %.3f, usable=%s",
             len(queries),
             len(merged),
             merged[0]["distance"] if merged else -1,
             usable)

    return {"hits": merged, "has_context": usable}


def _resolved(state):
    """The question with the implicit parts filled in, or the original if the
    understand node had nothing to resolve."""
    return state.get("resolved_question") or state["question"]


def generate(state):
    """Build the prompt from the retrieved chunks and call the model.

    Both forms of the question go into the prompt when they differ: the words
    the customer actually typed, and what those words mean given the
    conversation. The customer's own wording keeps the reply in their voice and
    register; the resolved form is what stops the model answering about every
    package in the context because the raw message named none of them.
    """
    user_prompt = build_user_prompt(
        state["question"], state["hits"], resolved=_resolved(state)
    )
    answer = _reply(state, SYSTEM_PROMPT, user_prompt)

    sources = [
        {
            "doc": hit["meta"]["doc"],
            "section": hit["meta"]["section"],
            "lang": hit["meta"]["lang"],
            "distance": round(hit["distance"], 3),
        }
        for hit in state["hits"]
    ]

    log.info("generate: answered in %d characters from %d sources",
             len(answer), len(sources))

    return {"answer": answer, "sources": sources}


def no_context(state):
    """Nothing relevant was found. Say so rather than letting the model guess.

    The model is still called, so the refusal comes back in the customer's own
    language rather than as a hardcoded English string.
    """
    log.info("no_context: nothing relevant for %r", state["question"][:60])

    # the resolved form, so a refusal to a follow-up names what was actually
    # being asked about rather than echoing a bare fragment back
    answer = _reply(state, NO_CONTEXT_PROMPT,
                    f"Customer question: {_resolved(state)}")

    return {"answer": answer, "sources": []}


def route_after_retrieval(state):
    """Decide which node runs next. The returned string is matched against the
    mapping defined in build.py."""
    return "generate" if state["has_context"] else "no_context"
