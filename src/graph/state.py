"""The state that flows through the agent graph.

Each node receives this, does one job, and returns the fields it changed.
LangGraph merges those back in before calling the next node, so this is the
shared workspace for the whole run.
"""

from typing import Callable, Optional, TypedDict


class ChatState(TypedDict, total=False):
    """total=False means every field is optional, so nodes can return only
    what they changed rather than rebuilding the whole state."""

    question: str         # what the customer asked, verbatim
    history: list         # recent turns, as {"role": ..., "content": ...}
    search_queries: list  # one or more standalone queries to retrieve on
    language: str         # 'ar' or 'en', detected from the question

    # The question with whatever the customer left implicit filled back in.
    #
    # "وبالعرض المضاعف؟" means nothing on its own; after the understand node
    # has read the previous turns it becomes a question about a named package.
    # Retrieval has always used this. Generation needs it too - without it the
    # model is shown a fragment alongside chunks about several packages and has
    # no way to tell which one the customer meant.
    #
    # Equal to question when there was nothing to resolve.
    resolved_question: str
    hits: list            # chunks retrieved from the vector store
    has_context: bool     # whether retrieval found anything usable
    answer: str           # the generated reply
    sources: list         # tidied source list for the interface

    # Where to send the reply as it is produced, one piece at a time.
    #
    # Set by an interface that wants to show the answer arriving rather than
    # waiting for it to finish. Left unset everywhere else - the scripts and
    # the tests want a finished string, and a node with no sink calls the model
    # in the ordinary blocking way.
    stream: Optional[Callable[[str], None]]
