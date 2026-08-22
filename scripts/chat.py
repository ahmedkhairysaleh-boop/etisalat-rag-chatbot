"""Talk to the chatbot from the command line.

    python -m scripts.chat

Type a question, get an answer. Blank line or 'exit' to quit.
"""

from src.graph.build import ask
from src.logging_config import setup_logging


def main():
    setup_logging("chat.log")

    print("e& Egypt assistant. Ask in Arabic or English. Type 'exit' to quit.\n")

    while True:
        question = input("> ").strip()

        if not question or question.lower() in {"exit", "quit"}:
            break

        result = ask(question)

        print(f"\n{result['answer']}\n")

        if result.get("sources"):
            print("sources:")
            for source in result["sources"]:
                print(f"  - {source['doc']} / {source['section'][:60]} "
                      f"({source['lang']}, {source['distance']})")
        print()


if __name__ == "__main__":
    main()