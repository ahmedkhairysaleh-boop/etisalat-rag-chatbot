"""The agent's nodes.

Each node takes the state, does one job, and returns only the fields it
changed. LangGraph merges the result into the state before the next node runs.
"""

import logging

from src.language import detect_language
from src.llm import generate as call_llm
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

# How many past turns the rewriter sees. More context rarely helps and every
# token counts against the rate limit.
HISTORY_TURNS = 4

# Multi-product questions are usually long enough to name both products, so
# short questions are not worth a decomposition call.
DECOMPOSE_LENGTH_MIN = 25

# Cap on merged results. Each query returns TOP_K hits, so without this a
# three-way split would triple the prompt and hit the tokens-per-minute limit.
MERGED_HIT_LIMIT = 8

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


def needs_rewrite(question, history):
    """Whether this question depends on the conversation around it."""
    return bool(history) and len(question.strip()) <= REWRITE_LENGTH_LIMIT


def understand(state):
    """Prepare the search queries.

    Two steps, each a separate model call so they can fail independently:

    1. Rewriting - a short follow-up like "what are its details" carries no
       product name, so it is rewritten using recent turns.
    2. Decomposition - a question naming two products needs one search per
       product. A single search returns chunks for whichever product matches
       more of the wording and misses the other entirely.

    Both are skipped when they cannot help, to save API calls.
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

    queries = [query]

    if len(query.strip()) >= DECOMPOSE_LENGTH_MIN:
        reply = call_llm(DECOMPOSE_PROMPT, f"Question: {query}")
        queries = parse_queries(reply, query)

        if len(queries) > 1:
            log.info("understand: split into %d searches: %s",
                     len(queries), queries)

    log.info("understand: language=%s, searching for %s", language, queries)

    return {"language": language, "search_queries": queries}


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

def generate(state):
    """Build the prompt from the retrieved chunks and call the model."""
    user_prompt = build_user_prompt(state["question"], state["hits"])
    answer = call_llm(SYSTEM_PROMPT, user_prompt)

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

    answer = call_llm(NO_CONTEXT_PROMPT,
                      f"Customer question: {state['question']}")

    return {"answer": answer, "sources": []}


def route_after_retrieval(state):
    """Decide which node runs next. The returned string is matched against the
    mapping defined in build.py."""
    return "generate" if state["has_context"] else "no_context"