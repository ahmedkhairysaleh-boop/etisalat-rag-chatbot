"""Central settings. Everything that might need changing lives here,
so no other file contains a hardcoded path or model name."""

from pathlib import Path

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "e& Knowledge base"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# --- vector store ---
COLLECTION_NAME = "etisalat_kb"

# multilingual: Arabic and English share one vector space, so an Arabic
# question can retrieve an English chunk. A monolingual model such as
# all-MiniLM-L6-v2 would silently return nonsense for Arabic.
EMBED_MODEL = "intfloat/multilingual-e5-base"

# --- chunking ---
MAX_CHARS = 1200      # target size of a chunk
OVERLAP = 150         # characters repeated between consecutive pieces
MIN_CHARS = 80        # ignore fragments smaller than this

# --- heading detection (PDF only) ---
# Numbered lines longer than this are Terms & Conditions list items, not
# section headings. Set from testing: at 90 the T&C clauses in HekayaMixat
# were each treated as a heading, producing 36 fake sections against the
# Arabic version's 15.
MAX_HEADING_LENGTH = 55

# --- retrieval ---
# Number of chunks returned per query. At 4, a question naming two products
# from different documents could retrieve chunks for only one of them.
TOP_K = 4