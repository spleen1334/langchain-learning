# RAG Overview

## What it is

Retrieval-Augmented Generation: fetch relevant text at query time and put it in the prompt, so the model answers from your data instead of its training set. Fixes the three core LLM limitations — stale knowledge, no access to private data, and hallucination — and gives you source attribution for free.

## The pipeline

```
Load → Split → Embed → Store → Retrieve → Augment prompt → Generate
```

Indexing (load/split/embed/store) happens offline; retrieval and generation happen per query.

### 1. Load
`Document(page_content, metadata)` is the universal unit. Loaders: `TextLoader`, `PyPDFLoader` (one Document per page, with page metadata), `WebBaseLoader` (bs4), `DirectoryLoader(glob=..., loader_cls=...)` — use `.lazy_load()` to stream large directories instead of materializing everything.

Put whatever you'll want to filter on into `metadata` at load time; retrofitting it later means re-indexing.

→ [`03_rag_and_memory/01_document_loaders.py`](../03_rag_and_memory/01_document_loaders.py)

### 2. Split
Chunking is the highest-leverage knob in the whole pipeline. Too large → the embedding averages several topics and matches nothing precisely; too small → retrieved text lacks the context needed to answer.

- `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap, separators=["\n\n","\n"," ",""])` — the default; tries to break on paragraph, then line, then word.
- `chunk_overlap` (~10–20% of chunk_size) prevents a fact from being severed at a boundary.
- `MarkdownHeaderTextSplitter` — splits on headers and promotes them into metadata, preserving document structure.
- `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` — code-aware boundaries (functions/classes).
- `split_documents(docs)` (vs `split_text`) keeps metadata on every chunk.

→ [`03_rag_and_memory/02_text_splitters.py`](../03_rag_and_memory/02_text_splitters.py)

### 3. Embed
An embedding maps text to a fixed-length vector where semantic similarity ≈ cosine similarity. `embed_query(str)` for the question, `embed_documents([str])` for the corpus (some models embed the two asymmetrically).

Model choice: `text-embedding-3-small` (1536 dims, ~$0.02/1M) is the sane default; `-3-large` (3072) when accuracy justifies 6× the price; `sentence-transformers/all-MiniLM-L6-v2` (384) or Ollama for local/private. **The query and the index must use the same model** — mixing them silently produces garbage.

`CacheBackedEmbeddings.from_bytes_store(underlying, store, namespace=...)` avoids re-paying for unchanged documents on re-index.

→ [`03_rag_and_memory/03_embeddings.py`](../03_rag_and_memory/03_embeddings.py), [`04_embeddings_deep.py`](../03_rag_and_memory/04_embeddings_deep.py)

### 4. Store
Vector stores index vectors for approximate nearest-neighbour search. This course uses **Chroma** (`langchain-chroma`); FAISS (local library), Pinecone/Qdrant (managed) and pgvector (Postgres extension) are the common alternatives.

Key operations:
- `Chroma.from_documents(docs, embedding, persist_directory=...)` — build and persist; reload with `Chroma(embedding_function=..., persist_directory=...)`
- `similarity_search(query, k=...)` and `similarity_search_with_score(...)` (Chroma returns a *distance* — smaller is better; convert if you want a similarity)
- `filter={"topic": "database"}` — metadata pre-filtering, often more effective than any retrieval tuning
- `as_retriever(search_type="similarity"|"mmr", search_kwargs={"k":3,"fetch_k":5})` — returns a Runnable that drops into a chain

**MMR** (Maximal Marginal Relevance) fetches `fetch_k` then selects `k` that are relevant *and* mutually diverse — the fix for "top 5 results are five copies of the same paragraph".

→ [`03_rag_and_memory/05_vector_stores.py`](../03_rag_and_memory/05_vector_stores.py)

### 5. Retrieve + generate
The canonical LCEL RAG chain:

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```
`RunnableParallel` (the dict literal) runs retrieval while passing the raw question through; `format_docs` joins page contents — include `doc.metadata["source"]` in that string if you want citations.

Prompt discipline matters as much as retrieval:
- "Answer based **only** on the following context" — grounding
- An explicit refusal instruction ("if the answer is not in the context, say …") — the fallback that stops hallucination on out-of-scope questions
- `with_structured_output(RAGResponse)` for answer + confidence + sources + follow-ups when downstream code needs to branch on confidence

→ [`03_rag_and_memory/06_rag_pipeline.py`](../03_rag_and_memory/06_rag_pipeline.py)

## Advanced retrieval

Plain top-k similarity fails in predictable ways; each strategy targets one failure mode.

| Strategy | API | Fixes |
|---|---|---|
| Multi-query | `MultiQueryRetriever.from_llm(retriever, llm)` | Vocabulary mismatch — LLM rewrites the question several ways, results deduped |
| Contextual compression | `ContextualCompressionRetriever(base_compressor=LLMChainExtractor.from_llm(llm), base_retriever=...)` | Context bloat — strips sentences in a chunk that don't bear on the question |
| Hybrid / ensemble | `EnsembleRetriever(retrievers=[BM25Retriever, vector], weights=[0.4, 0.6])` | Exact terms, acronyms, IDs, error codes that embeddings blur |
| Parent-document | `ParentDocumentRetriever(vectorstore, docstore, child_splitter, parent_splitter)` | Precision-vs-context tradeoff — match on small chunks, return the large parent |
| MMR | `as_retriever(search_type="mmr")` | Redundant near-duplicate results |

They compose: the repo's advanced chain stacks multi-query under compression. Each layer costs extra LLM calls and latency — measure before adopting.

→ [`03_rag_and_memory/07_advanced_rag.py`](../03_rag_and_memory/07_advanced_rag.py)

## Memory (the conversational half)

RAG answers one question; memory makes it a conversation. Strategies, in ascending cost:

- **Full history** — `RunnableWithMessageHistory(chain, get_session_history, input_messages_key, history_messages_key)` over a `MessagesPlaceholder("history")`. Simple, unbounded token growth. Per-user isolation via `session_id`.
- **Trimming** — `trim_messages(messages, max_tokens=..., strategy="last", token_counter=llm, include_system=True)`. Hard token ceiling.
- **Windowing** — keep the last k exchanges (custom `InMemoryChatMessageHistory` subclass). Predictable cost; older facts are simply lost.
- **Summary** — compress older turns into a running summary via a second LLM call, keep recent turns verbatim. Preserves facts at bounded cost; the demo shows name/city/job/pets all surviving.
- **Persistence** — `SQLChatMessageHistory(session_id, connection="sqlite:///...")` so history survives process restarts. (In LangGraph the equivalent is a checkpointer keyed by `thread_id`.)

Note that follow-up questions ("how does the second component work?") need history *before* retrieval to be resolvable — the research assistant handles this by passing history into the prompt alongside retrieved context.

→ [`03_rag_and_memory/08_conversation_memory.py`](../03_rag_and_memory/08_conversation_memory.py)

## Where each concept lives

| Concept | File |
|---|---|
| Loaders, `Document`, lazy loading | `03_rag_and_memory/01_document_loaders.py` |
| Chunk size/overlap, markdown & code splitting | `03_rag_and_memory/02_text_splitters.py` |
| Embedding providers and dimensions | `03_rag_and_memory/03_embeddings.py` |
| Cosine similarity, vector inspection, embedding cache | `03_rag_and_memory/04_embeddings_deep.py` |
| Chroma, scores, metadata filters, MMR, persistence | `03_rag_and_memory/05_vector_stores.py` |
| RAG chain, citations, refusal fallback, structured RAG | `03_rag_and_memory/06_rag_pipeline.py` |
| Multi-query, compression, BM25 hybrid, parent-document | `03_rag_and_memory/07_advanced_rag.py` |
| All memory strategies | `03_rag_and_memory/08_conversation_memory.py` |
| Full application combining RAG + memory + structured output | `projects/02_research_assistant.py` |
