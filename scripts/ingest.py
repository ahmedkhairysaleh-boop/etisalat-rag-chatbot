"""Build the vector database from the knowledge base documents.

Run once before using the chatbot:

    python -m scripts.ingest              # index everything
    python -m scripts.ingest --dry-run    # chunk only, show counts, do not embed
    python -m scripts.ingest --reset      # wipe and rebuild from scratch

Re-running is safe: chunk IDs are stable, so existing rows are overwritten
rather than duplicated.
"""

import argparse
import logging
import re

from src.config import KB_DIR
from src.extractors import sections_from_docx, sections_from_pdf
from src.chunking import build_chunks
from src.logging_config import setup_logging
from src.vectorstore import get_collection, reset_collection, add_chunks

log = logging.getLogger(__name__)


def clean_document_name(stem):
    """'Emerald (2)' -> 'Emerald'.

    The (2) is a download artifact. Stripping it means the Arabic and English
    versions of a document share one name in the metadata.
    """
    return re.sub(r"\s*\(\d+\)$", "", stem).strip()


def collect_all_chunks():
    """Read every document and return one flat list of chunks."""
    chunks = []

    # Arabic originals: .docx files in the knowledge base root
    for path in sorted(KB_DIR.glob("*.docx")):
        if path.name.startswith("~$"):      # Word lock file, not a document
            continue

        name = clean_document_name(path.stem)
        sections = sections_from_docx(path)
        found = build_chunks(sections, name, "ar", path.name)
        chunks += found

        log.info("%-18s ar  %3d sections -> %3d chunks",
                 name, len(sections), len(found))

    # English translations: .pdf files in the English subfolder
    for path in sorted((KB_DIR / "English").glob("*.pdf")):
        name = clean_document_name(path.stem)
        sections = sections_from_pdf(path)
        found = build_chunks(sections, name, "en", path.name)
        chunks += found

        log.info("%-18s en  %3d sections -> %3d chunks",
                 name, len(sections), len(found))

    return chunks


def summarise(chunks):
    """Print counts and sizes, so problems are visible before embedding."""
    documents = sorted(set(chunk.doc for chunk in chunks))
    log.info("total: %d chunks from %d documents", len(chunks), len(documents))

    for lang in ("ar", "en"):
        subset = [chunk for chunk in chunks if chunk.lang == lang]
        if not subset:
            continue
        lengths = [len(chunk.text) for chunk in subset]
        log.info("  %s: %3d chunks, %d-%d chars, average %d",
                 lang, len(subset), min(lengths), max(lengths),
                 sum(lengths) // len(lengths))


def main():
    parser = argparse.ArgumentParser(description="Index the knowledge base.")
    parser.add_argument("--reset", action="store_true",
                        help="delete the existing collection first")
    parser.add_argument("--dry-run", action="store_true",
                        help="chunk and report, but do not embed or store")
    args = parser.parse_args()

    setup_logging("ingest.log")

    if not KB_DIR.exists():
        raise SystemExit(f"knowledge base not found at {KB_DIR}")

    chunks = collect_all_chunks()
    summarise(chunks)

    if args.dry_run:
        log.info("dry run - nothing was embedded")
        return

    collection = reset_collection() if args.reset else get_collection()
    add_chunks(collection, chunks)

    log.info("done - collection holds %d chunks", collection.count())


if __name__ == "__main__":
    main()