# 03 — RAG and Memory

> **Numeric prefixes = suggested reading order**, and here it's a real pipeline dependency: load → split → embed → store → retrieve+generate (`01`–`06`). `07_advanced_rag.py` swaps in better retrievers on top of that pipeline; `08_conversation_memory.py` is independent of RAG and can be read any time.

The full retrieval-augmented generation pipeline — load, split, embed, store, retrieve, generate — plus conversation memory strategies for keeping context without blowing the token budget.

## `01_document_loaders.py`
- **What it does**: Ingestion. `TextLoader`, `WebBaseLoader` (bs4-backed HTML), `DirectoryLoader(glob=..., loader_cls=...)` with `.lazy_load()` for streaming large corpora, `PyPDFLoader` (run against `assets/sample_docs/langchain_demo.pdf`), and the anatomy of a `Document` (`page_content` + arbitrary `metadata` dict).
- **What it's for**: Step 1 of any RAG system — getting heterogeneous sources into a uniform `Document` shape, with metadata you can later filter on.

## `02_text_splitters.py`
- **What it does**: Chunking. `RecursiveCharacterTextSplitter` with explicit `separators` and chunk_size/overlap comparisons (200/500/1000), a side-by-side of overlap vs no-overlap showing boundary context loss, `MarkdownHeaderTextSplitter` (headers become metadata), `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` for code-aware splits, and `split_documents()` over loaded PDF pages to preserve metadata.
- **What it's for**: Chunk quality is the single biggest lever on retrieval quality — too big dilutes the embedding, too small loses context, and overlap is what keeps facts from being cut in half.

## `03_embeddings.py`
- **What it does**: Short reference file on embedding providers and their tradeoffs — OpenAI `text-embedding-3-small/large`/`ada-002` (dimensions and price per 1M tokens in a comment table), local `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2` (384 dims), and `OllamaEmbeddings` for fully local runs.
- **What it's for**: Picking an embedding model — cost vs accuracy vs "must not leave my machine".

## `04_embeddings_deep.py`
- **What it does**: What a vector actually is. `embed_query` vs `embed_documents`, vector dimension/norm inspection with numpy, hand-written cosine similarity ranking a small doc set against a query, and `CacheBackedEmbeddings.from_bytes_store(..., LocalFileStore, namespace=...)` proving a second embed call skips the API.
- **What it's for**: Demystifying semantic search, and cutting re-indexing cost — embedding caching is free money when you re-ingest the same corpus.

## `05_vector_stores.py`
- **What it does**: Chroma end to end. `Chroma.from_documents(..., persist_directory=...)`, `similarity_search`, `similarity_search_with_score` (with distance→similarity conversion), metadata filtering via `filter={"topic": "database"}`, `as_retriever(search_type="similarity"|"mmr", search_kwargs={"k", "fetch_k"})`, and a persist/reload cycle proving the index survives process restart.
- **What it's for**: The storage/query layer of RAG — plus MMR when top-k similarity keeps returning five near-duplicates instead of diverse context.

## `06_rag_pipeline.py`
- **What it does**: Assembling the RAG chain: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`. Four variants — basic; source-cited (docs formatted with `[i] source`); grounded with an explicit "I don't know" fallback for out-of-knowledge-base questions; and structured output via `with_structured_output(RAGResponse)` returning answer/confidence/sources/follow-up. Ends with a reusable `DocumentQA` class.
- **What it's for**: The reference implementation to copy — including the two things that matter in production: citing sources and refusing to answer when retrieval comes back empty.

## `07_advanced_rag.py`
- **What it does**: Retrieval strategies beyond top-k. `MultiQueryRetriever.from_llm` (LLM rewrites the query into several perspectives, results deduped); `ContextualCompressionRetriever` + `LLMChainExtractor` to strip irrelevant sentences from retrieved chunks; `EnsembleRetriever` hybrid search weighting `BM25Retriever` (keyword) against the vector retriever 0.4/0.6; `ParentDocumentRetriever` (small child chunks for precise matching, large parent chunks returned for context); and a combined multi-query + compression RAG chain.
- **What it's for**: Fixing real retrieval failures — vocabulary mismatch (multi-query), context bloat/token waste (compression), exact identifiers and acronyms that embeddings miss (BM25), and the precision-vs-context tradeoff (parent/child).

## `08_conversation_memory.py`
- **What it does**: Memory strategies. `RunnableWithMessageHistory` over `prompt | llm | parser` with `MessagesPlaceholder("history")` and a `get_session_history(session_id)` factory; per-user isolation via distinct `session_id`s; `trim_messages(strategy="last", max_tokens=..., include_system=True)`; a custom `WindowedChatHistory` subclass keeping only the last k exchanges (and demonstrating what gets forgotten); a running-summary memory that compresses old turns while keeping recent ones verbatim; and `SQLChatMessageHistory` persistence proven by rebuilding the chain from scratch and querying the raw SQLite table.
- **What it's for**: Bounded, predictable token cost in long conversations — and choosing between window (cheap, forgetful), summary (preserves facts, costs an extra LLM call), and full persistence (survives restarts).
