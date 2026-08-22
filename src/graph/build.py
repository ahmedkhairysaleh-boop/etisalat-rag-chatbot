"""Assemble the nodes into a LangGraph state machine.

Flow:

    understand ─┬─> smalltalk ─────────────────> END
                └─> retrieve ─┬─> generate ────> END
                              └─> no_context ──> END

Two conditional edges. The first sends greetings and thank-yous straight to a
fixed reply, skipping retrieval and the language model entirely. The second
decides whether the retrieved chunks are relevant enough to answer from, or
whether the customer should be told the documentation does not cover it.
"""

import logging

from langgraph.graph import StateGraph, END

from src.graph.state import ChatState
from src.graph.nodes import (
    understand,
    smalltalk,
    retrieve,
    generate,
    no_context,
    route_after_understand,
    route_after_retrieval,
)

log = logging.getLogger(__name__)

_graph = None


def build_graph():
    """Wire the nodes together and compile."""
    builder = StateGraph(ChatState)

    builder.add_node("understand", understand)
    builder.add_node("smalltalk", smalltalk)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("no_context", no_context)

    builder.set_entry_point("understand")

    # small talk never reaches the knowledge base
    builder.add_conditional_edges(
        "understand",
        route_after_understand,
        {
            "smalltalk": "smalltalk",
            "retrieve": "retrieve",
        },
    )

    # answer from context, or admit the documentation does not cover it
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "no_context": "no_context",
        },
    )

    builder.add_edge("smalltalk", END)
    builder.add_edge("generate", END)
    builder.add_edge("no_context", END)

    return builder.compile()


def get_graph():
    """Compile once and reuse."""
    global _graph

    if _graph is None:
        _graph = build_graph()
        log.info("agent graph compiled")

    return _graph


def ask(question, history=None):
    """Run one question through the agent.

    history is the recent conversation as a list of {"role", "content"} dicts.
    Passing it lets the understand node rewrite follow-up questions into ones
    that can be searched on their own.

    Returns the final state, which holds the answer, the sources used, the
    detected language, and whether any context was found.
    """
    graph = get_graph()

    return graph.invoke({
        "question": question,
        "history": history or [],
    })