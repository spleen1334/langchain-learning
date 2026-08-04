# Vector Database Vendor Options

Per-vendor detail for the alternatives introduced in `vector-databases-overview.md` (which has the theory and the comparison table). Chroma itself — the store this course's code uses — has its own doc: `vector-databases-chroma.md`.

## Qdrant

Open-source vector search engine written in Rust, runs as a server (Docker, self-hosted cluster, or Qdrant Cloud) — or in-memory for quick experiments via `QdrantClient(":memory:")`. Known for fast filtered search (payload filtering alongside vector search) and good support for on-disk indexes when a collection is too big for RAM.

```bash
pip install -qU langchain-qdrant
```
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

client = QdrantClient(":memory:")  # or url="http://localhost:6333" for a real server
vector_size = len(embeddings.embed_query("sample text"))

if not client.collection_exists("my_collection"):
    client.create_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(client=client, collection_name="my_collection", embedding=embeddings)
```
Reach for it when: you want production-grade self-hosting with strong metadata filtering, without committing to a fully managed SaaS.

## Pinecone

Fully managed, cloud-only vector database — there's no self-hosted mode. You create an index in Pinecone's dashboard/API, and LangChain talks to it over their client. No infrastructure to run or scale yourself.

```bash
pip install -qU langchain-pinecone
```
```python
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

pc = Pinecone(api_key="...")
index = pc.Index("my-index")  # index must already exist

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
```
Reach for it when: you want production scale with zero ops burden and are fine with a paid managed service and data leaving your infrastructure.

## Weaviate

Open-source, schema-driven vector database with a GraphQL-ish query layer and first-class **hybrid search** (combining dense vector similarity with sparse/keyword (BM25) scoring in one query). Runs self-hosted (Docker) or as Weaviate Cloud.

```bash
pip install -qU langchain-weaviate
```
```python
import weaviate
from langchain_weaviate import WeaviateVectorStore

client = weaviate.connect_to_local()  # or connect_to_weaviate_cloud(...)
vector_store = WeaviateVectorStore.from_documents(docs, embeddings, client=client)
```
Reach for it when: hybrid search (keyword + semantic) matters, or you want schema/class-based data modeling built into the store itself.

## FAISS (Facebook AI Similarity Search)

Not a database — a **library** for efficient similarity search over vectors, running entirely in-process (no server, no client/server protocol). LangChain's `FAISS` vector store wraps it with `.save_local()` / `FAISS.load_local()` for manual persistence to disk.

```bash
pip install -qU langchain-community faiss-cpu   # or faiss-gpu
```
```python
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(docs, embeddings)
vector_store.save_local("./faiss_index")

reloaded = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
```
Reach for it when: you want the fastest possible in-memory search with no server overhead at all, e.g. a batch job or an app that loads a prebuilt index at startup. Trade-off: you own persistence, scaling, and concurrent-write handling yourself — there's no server managing that for you.

## How to choose

- **Already prototyping locally, no infra** → Chroma (what this course uses) or FAISS.
- **Need it in production but want to self-host** → Qdrant or Weaviate.
- **Want managed, zero-ops, willing to pay and send data to a SaaS** → Pinecone (or Qdrant/Weaviate Cloud).
- **Need hybrid keyword+vector search out of the box** → Weaviate (or Qdrant, which also supports sparse vectors).
- **Need raw speed / full control, comfortable managing persistence yourself** → FAISS.

All five share LangChain's `VectorStore` interface (`from_documents`, `similarity_search`, `as_retriever`, ...), so swapping between them later is mostly a constructor change, not a rewrite of retrieval logic.
