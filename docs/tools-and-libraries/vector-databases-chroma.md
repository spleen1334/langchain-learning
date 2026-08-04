# Chroma Overview

## What it is

Chroma is an **open-source embedding database (vector store)**: it stores embedding vectors alongside their source text and metadata, and lets you query by semantic similarity instead of exact match. Under the hood a similarity search is a nearest-neighbor lookup over vectors — Chroma indexes them so that lookup stays fast as the collection grows.

It's used in this course as the vector store behind RAG pipelines: documents get embedded (see `pydantic-overview.md`'s neighbor, the embeddings scripts in `03_rag_and_memory/`), stored in Chroma, and retrieved by similarity to a query at answer time.

## Core concepts

### Creating a store from documents
```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=docs,               # list[Document]
    embedding=embeddings_model,   # any LangChain Embeddings instance
    persist_directory="./chroma_db",  # omit for an ephemeral in-memory store
)
```
- `from_documents` embeds every document and writes it into a collection in one call.
- No `persist_directory` → in-memory only, discarded when the process exits.
- A `collection_name` can be passed to keep multiple logical stores in the same directory.

### Reconnecting to an existing store
```python
vectorstore = Chroma(
    embedding_function=embeddings_model,
    persist_directory="./chroma_db",
)
```
Reopening with the same `persist_directory` (and the same embedding model — dimensions must match) gives you back the previously indexed data without re-embedding anything.

### Similarity search
```python
results = vectorstore.similarity_search(query, k=3)
```
- Returns the `k` most similar `Document`s.
- `similarity_search_with_score(query, k=3)` also returns a score. **Chroma's default score is a distance (lower = more similar), not a similarity** — invert it (e.g. `1 / (1 + score)`) if you want higher-is-better. See "Distance vs. similarity" in `vector-databases-overview.md` for why, and what `05_vector_stores.py` does with it.
- Metadata filtering: `vectorstore.similarity_search(query, k=3, filter={"topic": "database"})`.

### Retrievers — is `as_retriever` the same thing as `similarity_search`?
Underneath, yes: by default `as_retriever` just calls `similarity_search` for you. It's not a different search — it's a **different interface** around the same store, built for a different situation.

```python
# Direct call — you're driving
docs = vectorstore.similarity_search(query, k=3)

# Retriever — a Runnable wrapping the same call
retriever = vectorstore.as_retriever(
    search_type="similarity",   # or "mmr", "similarity_score_threshold"
    search_kwargs={"k": 3},
)
docs = retriever.invoke(query)   # same result as above
```

What `as_retriever` actually buys you:
- **A `Runnable`, not a one-off method call.** That means `.invoke()`, `.batch()`, and — critically — it can be piped with `|` into an LCEL chain or dropped straight into a LangGraph node, the same shape as every other LangChain component. `similarity_search` is just a plain Python method; you'd have to wrap it yourself to compose it that way.
- **The search strategy becomes configuration, not code.** `search_type="mmr"` or `"similarity_score_threshold"` swaps the retrieval *algorithm* (see "Beyond plain top-k: MMR" in `vector-databases-overview.md`) without changing a single line of calling code — useful because a chain built against a `retriever.invoke(query)` interface doesn't care what's happening inside it.

**Rule of thumb:** reaching for the vector store directly (`similarity_search`, `similarity_search_with_score`) in a script or notebook where you're just inspecting results is fine and simpler. Reach for `as_retriever` the moment that search needs to slot into a chain, an agent, or anything else expecting a `Runnable`.

### Persistence
Chroma with a `persist_directory` writes to disk automatically — there's no separate "save" call to remember. Point a new `Chroma(...)` instance at the same directory (and same embedding model) later to pick up where you left off.

## Why LangChain uses it here
- **Local, zero-infra RAG** — no external service to run or pay for, good fit for a course/dev environment.
- **Drop-in `VectorStore` interface** — `similarity_search`, `as_retriever`, `add_documents` are the same shape as every other LangChain vector store, so swapping in Pinecone/pgvector/FAISS later doesn't change calling code.
- **Metadata filtering** — lets retrieval be scoped (by source, topic, date, etc.) without a separate database.

## Most common cases in practice
| Use case | How |
|---|---|
| Index a batch of documents | `Chroma.from_documents(documents=docs, embedding=embeddings_model, persist_directory=...)` |
| Reopen a previously persisted store | `Chroma(embedding_function=embeddings_model, persist_directory=...)` |
| Plain top-k retrieval | `vectorstore.similarity_search(query, k=k)` |
| Retrieval with relevance scores | `vectorstore.similarity_search_with_score(query, k=k)` (remember: distance, not similarity) |
| Diverse (non-redundant) retrieval | `vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})` |
| Scope retrieval to a subset | `similarity_search(query, k=k, filter={"field": "value"})` |
| Wire into an LCEL/LangGraph chain | `vectorstore.as_retriever(search_kwargs={"k": k})` |

## Chroma vs. the competition

See `vector-databases-overview.md` for the full field (Qdrant, Pinecone, Weaviate, FAISS). This section is the "why Chroma, specifically" case.

### What Chroma does well
- **Lowest friction to first result.** `pip install`, `Chroma.from_documents(...)`, done — no server to run, no account to create, no schema to design up front. FAISS is the only other option this frictionless, and FAISS isn't a database (no metadata store, no filtering, you own persistence by hand).
- **Persistence without ops.** A `persist_directory` gets you durable storage across process restarts with zero server to manage — Qdrant/Weaviate/Pinecone all require standing up (or paying for) a server to get that.
- **Metadata filtering out of the box**, unlike FAISS where you'd build that yourself alongside the index.
- **Good default embedded story for local dev, notebooks, and small-to-mid-size apps** — exactly the profile of the code in this course.

### Why choose it over the others
- Over **FAISS**: you get metadata + filtering + a real persistence story for free, at a modest speed cost — FAISS is a bare similarity-search library, Chroma is a database built around one.
- Over **Qdrant / Weaviate**: no server to deploy, configure, or keep running — big win for local dev and small projects; the trade is you give up their production-scale features (see below).
- Over **Pinecone**: self-hosted and free, your data never leaves your machine, no vendor account or billing — at the cost of Pinecone's fully-managed scaling.

### Where it's weaker
- **Scale.** Chroma is not built for massive (100M+ vector), high-QPS, horizontally-sharded workloads the way Qdrant, Weaviate, and Pinecone are — those were designed for production scale from the start.
- **Filtering sophistication.** Qdrant in particular is known for very fast, expressive payload filtering combined with vector search on huge collections; Chroma's filtering is simpler.
- **Hybrid search.** Weaviate has first-class hybrid (dense + BM25) search built in; Chroma doesn't natively — you'd combine it with a separate keyword search yourself.
- **Managed/production tooling.** Pinecone (and Qdrant/Weaviate Cloud) hand you monitoring, scaling, backups, and multi-region as a service; a self-run Chroma instance means you're responsible for all of that, though Chroma Cloud narrows this gap.
- **Raw speed at very large scale.** FAISS, being a bare library with no database overhead, will typically out-run Chroma on pure similarity-search throughput once metadata/filtering aren't needed.

### Bottom line
Chroma trades the production-scale muscle of Qdrant/Weaviate/Pinecone and the raw speed of FAISS for near-zero setup and ops — which is exactly why it's the right default for this course's RAG examples, and a reasonable choice for real small-to-mid projects. Migrating off it later is cheap because of LangChain's shared `VectorStore` interface, so starting here isn't a trap.

## Where it's used in this repo
- `03_rag_and_memory/05_vector_stores.py` — Chroma basics, scored search, metadata filtering, `as_retriever`, MMR, and persistence, one function per concept.
- `03_rag_and_memory/06_rag_pipeline.py`, `03_rag_and_memory/07_advanced_rag.py` — Chroma wired into full RAG chains (multi-query, parent-document, hybrid retrieval).
- `projects/02_research_assistant.py` — a persisted Chroma store (`./research_db`) as the backing knowledge base for a longer-lived assistant.
