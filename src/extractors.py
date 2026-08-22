"""Turn each source document into a list of (heading, text) sections.

Two formats need different handling:
  - the Arabic .docx originals carry real Word heading styles
  - the English .pdf translations lose all styling, so headings are detected
    by pattern instead
"""

import re

from docx import Document
from pypdf import PdfReader

from src.config import MAX_HEADING_LENGTH


def sections_from_docx(path):
    """Arabic originals. Word stores heading styles, so section boundaries are
    explicit rather than guessed."""
    document = Document(path)
    sections = []
    heading_path = {}        # heading level -> text
    heading, buffer = "", []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text or text == ".":
            continue

        style = paragraph.style.name or ""
        match = re.match(r"Heading (\d+)", style)

        if match:
            if buffer:
                sections.append((heading, "\n".join(buffer)))
                buffer = []

            level = int(match.group(1))
            heading_path[level] = text

            # forget any deeper headings left over from the previous branch
            for deeper in [lvl for lvl in heading_path if lvl > level]:
                del heading_path[deeper]

            # "7. Terms & Conditions > Family line" keeps the parent topic
            # attached to a generic subheading like "Family line"
            heading = " > ".join(heading_path[lvl] for lvl in sorted(heading_path))
        else:
            buffer.append(text)

    if buffer:
        sections.append((heading, "\n".join(buffer)))

    return sections


HEADING_PATTERN = re.compile(r"^\d+(\.\d+)*[\.\)]?\s+\S")


def sections_from_pdf(path):
    """English translations. No style information survives in a PDF, so a line
    is treated as a heading when it starts with a number, is short, and does
    not end like a sentence."""
    reader = PdfReader(path)
    raw_text = "\n".join((page.extract_text() or "") for page in reader.pages)

    sections = []
    heading, buffer = "", []

    for line in raw_text.split("\n"):
        text = line.strip()
        if not text:
            continue

        is_heading = (
            HEADING_PATTERN.match(text)
            and len(text) < MAX_HEADING_LENGTH
            and not text.endswith((".", ",", ":", ";"))
        )

        if is_heading:
            if buffer:
                sections.append((heading, "\n".join(buffer)))
                buffer = []
            heading = text
        else:
            buffer.append(text)

    if buffer:
        sections.append((heading, "\n".join(buffer)))

    return sections