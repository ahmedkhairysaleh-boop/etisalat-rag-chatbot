"""Central settings. Everything that might need changing lives here,
so no other file contains a hardcoded path or model name.

This module also sets the environment variables that control how the embedding
model is loaded. They have to be in place before sentence-transformers is
imported, and every other module imports this one first, so here is the only
place they reliably take effect.
"""

import os
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

# the model runs locally and the knowledge base is small, so there is nothing
# for a GPU to do. Naming the device stops torch probing for CUDA on every
# load, which costs seconds on a machine that has none.
EMBED_DEVICE = "cpu"

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
# Number of chunks returned per query.
#
# Raised from 4 to 6 after two failures traced to the same cause. Asking for
# the cheapest way to top up minutes on Hekaya Internet 46 filled all four
# slots with chunks about plan 46 and never reached the extra-unit plans that
# answer it. Asking the price of Aqwa Card with no tier named filled them with
# whatever was nearest, and the closest chunk belonged to a different product
# entirely.
#
# Both are the same shape: a product described across many similar chunks
# crowds out the one chunk that actually answers the question. Six gives the
# right chunk room to appear without much cost - the merged results are capped
# at MERGED_HIT_LIMIT in the retrieve node, so the prompt cannot grow without
# bound when a question is split into several searches.
TOP_K = 6


# --- model loading ---

def _hub_cache_dir():
    """Where HuggingFace keeps downloaded models on this machine."""
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.getenv("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_is_cached():
    """Whether the embedding model has already been downloaded.

    A cached model has a directory named after it with at least one snapshot
    inside. An empty or half-written directory does not count - treating that
    as cached would send the loader offline with nothing to load.
    """
    folder = _hub_cache_dir() / f"models--{EMBED_MODEL.replace('/', '--')}"
    snapshots = folder / "snapshots"

    return snapshots.is_dir() and any(snapshots.iterdir())


def configure_model_loading():
    """Keep the embedding model load off the network where possible.

    sentence-transformers contacts HuggingFace on every load to check whether
    the model has changed upstream. The check runs before the model is
    returned, so on a slow or filtered connection the app sits on a blank page
    waiting for it - and it can only ever confirm what is already on disk.

    Going offline is conditional on the model actually being cached, so a
    fresh clone on another machine still downloads it on first run rather than
    failing with a confusing offline error.

    setdefault throughout: an explicitly set variable in the shell or in .env
    is a deliberate choice and is left alone.
    """
    if _model_is_cached():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # the tokenizer runs one query at a time here, so its worker pool would
    # only add fork warnings and start-up cost
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


configure_model_loading()
