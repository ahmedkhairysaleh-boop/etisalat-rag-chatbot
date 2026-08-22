"""Search the knowledge base from the command line.

A debugging tool, not part of the chatbot. It shows which chunks a question
retrieves and how close each one is, which is how you tell a retrieval problem
from a prompting problem.

    python -m scripts.ask "How much does Emerald 430 cost?"
    python -m scripts.ask "كام سعر اميرالد 430؟"
"""

import sys

from src.language import detect_language
from src.vectorstore import get_collection, search


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m scripts.ask "your question"')

    question = " ".join(sys.argv[1:])
    collection = get_collection()

    print(f"\nquestion: {question}")
    print(f"language: {detect_language(question)}")
    print(f"indexed:  {collection.count()} chunks\n")

    for position, hit in enumerate(search(collection, question), start=1):
        meta = hit["meta"]
        print(f"--- {position}. distance {hit['distance']:.3f}  "
              f"[{meta['lang']}] {meta['doc']}")
        print(f"    section: {meta['section'][:70]}")
        preview = hit["text"].replace("\n", " ")[:200]
        print(f"    {preview}...\n")


if __name__ == "__main__":
    main()