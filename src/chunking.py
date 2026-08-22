"""Turn extracted sections into the chunks that get embedded.

Three jobs:
  - prefix every chunk with its heading, so a package name travels with its price
  - split sections longer than MAX_CHARS at paragraph boundaries, with overlap
  - drop fragments too short to be useful
"""

from dataclasses import dataclass

from src.config import MAX_CHARS, OVERLAP, MIN_CHARS


@dataclass
class Chunk:
    """One indexable piece of the knowledge base."""
    text: str         # heading + body, this is what gets embedded
    doc: str          # Emerald, DataLine, HekayaInternet, ...
    lang: str         # 'ar' or 'en'
    section: str      # heading breadcrumb
    source: str       # file it came from


def split_long_text(text):
    """Break an oversized section on paragraph boundaries. Consecutive pieces
    share OVERLAP characters so a fact sitting on the seam is not lost."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    pieces, current = [], ""

    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > MAX_CHARS:
            pieces.append(current)
            current = current[-OVERLAP:] + "\n" + paragraph
        else:
            current = f"{current}\n{paragraph}" if current else paragraph

    if current:
        pieces.append(current)

    return pieces


def build_chunks(sections, doc, lang, source):
    """Sections -> chunks, with the heading prefixed onto each one.

    Without the prefix a chunk reading 'Monthly price: 430 EGP' loses the
    package name that sits in the heading above it, and a search for
    'Emerald 430 price' would not match even though the number is right there.
    """
    chunks = []

    for heading, body in sections:
        for piece in split_long_text(body):
            text = f"{heading}\n{piece}" if heading else piece
            text = text.strip()

            # measure the finished chunk, not the body alone: a glossary entry
            # like "*811#" has a short body but the heading carries the code,
            # and together they are worth indexing
            if len(text) < MIN_CHARS:
                continue

            chunks.append(Chunk(
                text=text,
                doc=doc,
                lang=lang,
                section=heading or "(intro)",
                source=source,
            ))

    return chunks