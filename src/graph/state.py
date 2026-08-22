"""The state that flows through the agent graph.

Each node receives this, does one job, and returns the fields it changed.
LangGraph merges those back in before calling the next node, so this is the
shared workspace for the whole run.
"""

from typing import TypedDict


class ChatState(TypedDict, total=False):
    """total=False means every field is optional, so nodes can return only
    what they changed rather than rebuilding the whole state."""

    question: str        # what the customer asked, verbatim
    history: list        # recent turns, as {"role": ..., "content": ...}
    search_queries: list  # one or more standalone queries to retrieve on    language: str        # 'ar' or 'en', detected from the question
    hits: list           # chunks retrieved from the vector store
    has_context: bool    # whether retrieval found anything usable
    answer: str          # the generated reply
    sources: list        # tidied source list for the interface