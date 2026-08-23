"""Assemble the nodes into a LangGraph state machine.

Flow:

    understand ─┬─> smalltalk ─────────────────> END
                └─> retrieve ─┬─> generate ────> END
                              └─> no_context ──> END

Two conditional edges. The first sends greetings and thank-yous straight to a
fixed reply, skipping retrieval and the language model entirely. The second
decides whether the retrieved chunks are relevant enough to answer from, or
whether the customer should be told the documentation does not cover it.

Two ways in. ask() runs the graph and returns the finished state - what the
scripts, the tests and the Chainlit interface want. AnswerStream runs the same
graph on a background thread and hands back the reply piece by piece as the
model writes it.
"""

import logging
import threading
from queue import Queue

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

# put on the queue when the run is over. A unique object rather than None or a
# sentinel string, so it can never be confused with a piece of the answer.
_DONE = object()


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


class AnswerStream:
    """One question, answered piece by piece.

    The graph has to run start to finish to produce a reply, and the reply is
    written inside a node - so there is no way to both run the graph and
    receive the text as it appears from a single thread. The graph runs on its
    own thread instead, pushing each piece into a queue, and the caller reads
    the queue.

    Usage:

        stream = AnswerStream(question, history)

        for piece in stream.tokens():
            ...                     # paint it

        stream.result["sources"]    # available once tokens() is exhausted

    The pieces are only the customer-facing reply. The rewriting and
    decomposition calls inside the understand node are not streamed, so nothing
    the agent says to itself reaches the screen.
    """

    def __init__(self, question, history=None):
        self._queue = Queue()
        self._outcome = {}

        self.result = None

        self._thread = threading.Thread(
            target=self._run,
            args=(question, history or []),
            name="answer-stream",
            daemon=True,
        )

    def _run(self, question, history):
        """The graph run. Whatever happens, _DONE goes on the queue - without
        it a failure would leave the reader waiting forever."""
        try:
            self._outcome["state"] = get_graph().invoke({
                "question": question,
                "history": history,
                "stream": self._queue.put,
            })
        except Exception as error:          # noqa: BLE001 - re-raised below
            self._outcome["error"] = error
        finally:
            self._queue.put(_DONE)

    def tokens(self):
        """Yield the reply as it is written.

        Anything the graph raised is re-raised here, on the caller's thread, so
        an interface can handle a rate limit the same way it always has.
        """
        self._thread.start()

        while True:
            piece = self._queue.get()

            if piece is _DONE:
                break

            yield piece

        self._thread.join()

        error = self._outcome.get("error")

        if error is not None:
            raise error if isinstance(error, RuntimeError) else RuntimeError(
                f"the agent failed: {error}"
            )

        self.result = self._outcome.get("state", {})
