"""ChromaDB storage and retrieval.

The only module that knows which vector database is in use. Everything else
talks to it through these four functions, so swapping Chroma for something
else would change this file and nothing more.

The open collection is held at module level. Loading the embedding model takes
seconds, and the retrieve node asks for the collection on every question, so
without this the client would be rebuilt on each turn.
"""

import logging

import chromadb
from chromadb.utils import embedding_functions

from src.config import (CHROMA_DIR, COLLECTION_NAME, EMBED_DEVICE,
                        EMBED_MODEL, TOP_K)
from src.language import detect_language, normalize_query, other_language

log = logging.getLogger(__name__)

BATCH_SIZE = 100

# above this cosine distance a result is too weak to trust, and the search
# falls back to the other language
WEAK_MATCH_DISTANCE = 0.45

_collection = None


def _embedding_function():
    """The model that turns text into vectors. Runs locally, so there is no
    API cost and no rate limit."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device=EMBED_DEVICE,
    )


def _open_collection():
    """Open the collection, creating it if this is the first run.

    PersistentClient writes to disk, so the embeddings survive restarts and
    the documents are embedded once rather than on every launch.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def get_collection():
    """The open collection, opening it on first use.

    Every caller shares one collection and therefore one loaded copy of the
    embedding model.
    """
    global _collection

    if _collection is None:
        _collection = _open_collection()
        log.info("collection ready: %s", COLLECTION_NAME)

    return _collection


def reset_collection():
    """Delete the collection and start over."""
    global _collection

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
        log.info("deleted existing collection")
    except Exception:
        log.info("no existing collection to delete")

    # the cached handle now points at something that no longer exists
    _collection = None

    return get_collection()


def add_chunks(collection, chunks):
    """Embed and store chunks, in batches.

    IDs are built from the document, language and position, so re-running the
    ingest overwrites the same rows instead of creating duplicates. That makes
    recovering from an interrupted run a matter of running it again.
    """
    ids = [f"{chunk.doc}-{chunk.lang}-{index}"
           for index, chunk in enumerate(chunks)]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "doc": chunk.doc,
            "lang": chunk.lang,
            "section": chunk.section,
            "source": chunk.source,
        }
        for chunk in chunks
    ]

    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        log.info("indexed %d/%d chunks", min(end, len(chunks)), len(chunks))


def _query(collection, question, k, lang):
    """One search, restricted to a single language."""
    result = collection.query(
        query_texts=[question],
        n_results=k,
        where={"lang": lang},
    )

    return [
        {"text": text, "meta": meta, "distance": distance}
        for text, meta, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def search(collection, question, k=TOP_K):
    """Retrieve the k chunks closest to the question.

    Results are filtered to the language the question was asked in. The Arabic
    and English documents cover the same material, so searching both at once
    returns near-duplicate chunks and wastes half the context window. Filtering
    also means the model reads its sources in the language it is about to
    answer in, which reduces mistakes on numbers and package names.

    If the best match is weak, the other language is searched as a fallback -
    a topic may be covered better in one version than the other.

    Arabic numerals in the question are rewritten as Western ones before the
    search. Every document writes its numbers with Western digits, so a
    customer asking about "حكاية ميكسات ١٨٠" would otherwise miss the chunk
    naming it "حكاية ميكسات 180" - and since the six bundle chunks are
    identical apart from that number, missing it leaves retrieval with nothing
    to separate them by. The language is detected from the original text,
    before the rewrite, so nothing about the answer's language changes.
    """
    lang = detect_language(question)
    query = normalize_query(question)

    hits = _query(collection, query, k, lang)

    if not hits or hits[0]["distance"] > WEAK_MATCH_DISTANCE:
        log.info("weak match in '%s', falling back to '%s'",
                 lang, other_language(lang))
        hits += _query(collection, query, k, other_language(lang))
        hits.sort(key=lambda hit: hit["distance"])
        hits = hits[:k]

    return hits
