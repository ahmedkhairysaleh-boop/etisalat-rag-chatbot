"""Time each stage of start-up, to find where the 30 seconds actually goes.

    python scripts/profile_startup.py

Reads nothing and writes nothing. Run it twice: the first run includes reading
the model off disk, the second usually finds it in the operating system's file
cache, and the gap between the two tells you whether the disk is the problem.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os


class Stage:
    """Time one stage and print it as soon as it finishes, so a stage that
    hangs is visible while it is hanging rather than after."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        print(f"  {self.label:<38}", end="", flush=True)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_):
        print(f"{time.perf_counter() - self.started:6.2f}s")


def main():
    total = time.perf_counter()

    print("\nenvironment")

    from src.config import (CHROMA_DIR, COLLECTION_NAME, EMBED_DEVICE,
                            EMBED_MODEL, _hub_cache_dir, _model_is_cached)

    cache = _hub_cache_dir()
    folder = cache / f"models--{EMBED_MODEL.replace('/', '--')}"

    print(f"  model            {EMBED_MODEL}")
    print(f"  device           {EMBED_DEVICE}")
    print(f"  hub cache        {cache}")
    print(f"  cache exists     {cache.is_dir()}")
    print(f"  model folder     {folder}")
    print(f"  model cached     {_model_is_cached()}")
    print(f"  HF_HUB_OFFLINE   {os.environ.get('HF_HUB_OFFLINE', '(unset)')}")
    print(f"  TRANSFORMERS_OFF {os.environ.get('TRANSFORMERS_OFFLINE', '(unset)')}")

    if folder.is_dir():
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
        print(f"  model on disk    {size / 1e6:.0f} MB")

    print("\nstages")

    with Stage("import torch"):
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    with Stage("import sentence_transformers"):
        from sentence_transformers import SentenceTransformer

    with Stage("load embedding model"):
        model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

    with Stage("first encode (warm-up)"):
        model.encode(["كام سعر الباقة؟"])

    with Stage("second encode"):
        model.encode(["how much is the package?"])

    with Stage("import chromadb"):
        import chromadb

    with Stage("open persistent client"):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    with Stage("get collection"):
        from src.vectorstore import get_collection
        collection = get_collection()

    with Stage("count chunks"):
        count = collection.count()

    with Stage("one search"):
        collection.query(query_texts=["Emerald 430 price"], n_results=4)

    print(f"\n  {'TOTAL':<38}{time.perf_counter() - total:6.2f}s")
    print(f"  {count} chunks indexed\n")


if __name__ == "__main__":
    main()
