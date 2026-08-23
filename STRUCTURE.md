# Project structure

What each file is responsible for, and why the boundaries fall where they do.
The organising rule is one concern per file: any single change should land in
one place.

```
etisalat-chatbot/
├── app.py                    Streamlit chat interface
├── requirements.txt
├── .env                      provider, model and API key (not committed)
├── .streamlit/config.toml    telemetry off, file watcher settings
│
├── src/
│   ├── config.py             every setting and path
│   ├── language.py           Arabic/English detection, query normalisation
│   ├── logging_config.py     log format and file handler
│   ├── extractors.py         .docx and .pdf to text with headings
│   ├── chunking.py           text to chunks, split on headings
│   ├── vectorstore.py        ChromaDB: store and search
│   ├── llm.py                the language model client
│   ├── prompts.py            every prompt string, and the formatters
│   └── graph/
│       ├── state.py          the state that flows between nodes
│       ├── nodes.py          what each node does
│       └── build.py          wiring, plus ask() and AnswerStream
│
├── scripts/
│   ├── ingest.py             read documents, chunk, embed, store
│   ├── chat.py               terminal chat
│   ├── ask.py                one question, one answer
│   ├── evaluate.py           run the test questions, write a report
│   └── profile_startup.py    time each stage of start-up
│
├── tests/questions.py        bilingual test cases
├── assets/                   the four bot avatars
├── e& Knowledge base/        source documents (not committed)
├── chroma_db/                the index (not committed, rebuilt by ingest)
└── logs/                     application log
```

## The interface

**`app.py`** — the Streamlit chat interface, and the only file that knows
anything about how the conversation looks. Renders bubbles as custom HTML
rather than `st.chat_message` so alignment, colour and avatars are all
controllable, and so Arabic can be laid out right-to-left.

Three pieces of timing work live here. The embedding model loads on a
background thread, because doing it before the first render left the page
blank for fifteen seconds. A question is handled across two reruns, so the
customer's own message paints before the agent is asked anything. And the
answer is painted as it is written rather than when it is finished.

It also protects service codes from the markdown converter, which reads the
asterisks in `*319*155#` as emphasis and would otherwise leave `319155#` on
screen.

## Settings

**`src/config.py`** — paths, model names, chunk sizes, retrieval limits.
Nothing else in the project contains a hardcoded path or model name, so
retuning is done here and nowhere else.

It also sets the environment variables that control how the embedding model
loads. Those have to be in place before sentence-transformers is imported, and
every other module imports this one first, so this is the only place they
reliably take effect.

## Reading the documents

**`src/extractors.py`** — turns `.docx` and `.pdf` files into text while
keeping track of which heading each passage sits under. The two formats need
different heading detection: Word carries real heading styles, a PDF only has
lines that look like headings.

**`src/chunking.py`** — cuts that text into retrievable pieces at the
documents' own headings, prefixing the heading breadcrumb onto every chunk so a
piece still says what it is about once it is retrieved on its own. Long
sections are split further with an overlap.

## Storage and retrieval

**`src/vectorstore.py`** — the only module that knows ChromaDB is in use.
Everything else goes through its four functions, so swapping the vector
database would change this file and nothing more.

Search is filtered to the language the question was asked in, since the Arabic
and English documents cover the same material and searching both returns
near-duplicates. If the best match is weak it falls back to the other language.

**`src/language.py`** — decides whether text is Arabic or English, and
normalises a question before it is embedded. Arabic-Indic digits become Western
ones, because the documents write every number in Western digits and `١٨٠`
and `180` are unrelated tokens to the embedding model. Invisible characters —
narrow no-break spaces, non-breaking hyphens — are flattened for the same
reason. Only the search query is touched; the customer's message is displayed
and answered exactly as written.

## The model

**`src/llm.py`** — the only module that knows which provider is in use.
Providers are reached through their OpenAI-compatible endpoints, so moving
between Groq, Gemini, Qwen or Mistral means editing `.env` and nothing else.

Two ways to call it: `generate()` waits for the whole reply, `generate_stream()`
yields it in pieces. Both share one request builder so they cannot drift apart.
Token usage is logged per call, which is what makes a rate limit diagnosable
rather than mysterious.

**`src/prompts.py`** — every prompt string in the project, plus the functions
that assemble them. Separate from the agent logic because prompts change far
more often than code does during testing.

## The agent

**`src/graph/state.py`** — the shape of the state that flows between nodes.
Each node returns only the fields it changed.

**`src/graph/nodes.py`** — what each node does, and the thresholds that decide
routing. Also the three gates that keep cost down: whether a question needs
rewriting, whether it needs decomposing, and whether the retrieved chunks are
relevant enough to answer from. Each gate exists because the alternative was
spending an API call to be told nothing.

**`src/graph/build.py`** — wires the nodes into the state machine and exposes
two ways in. `ask()` runs the graph and returns the finished state, which is
what the scripts and tests want. `AnswerStream` runs the same graph on a
background thread and hands the reply back piece by piece as the model writes
it.

```
understand ─┬─> smalltalk ─────────────────> END
            └─> retrieve ─┬─> generate ────> END
                          └─> no_context ──> END
```

## Scripts

**`ingest.py`** reads the knowledge base, chunks it, embeds it and stores it.
Run once, and again whenever the documents or the chunking change. IDs are
derived from document, language and position, so re-running overwrites rather
than duplicating — an interrupted run is recovered by running it again.

**`chat.py`** is a terminal conversation, **`ask.py`** answers a single
question, **`evaluate.py`** runs the test set and writes a report, and
**`profile_startup.py`** times each stage of start-up.

## What is not committed

`.env` holds the API key. `chroma_db/` is rebuilt by the ingest script. The
knowledge base is the company's own documents. `logs/` is local. All are
listed in `.gitignore`.
