import numpy as np
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from ollama import embeddings

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def basic_embeddings():

    # single text
    text = "What is Machine Learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")


def batch_embeddings():
    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]

    # embed_documents sends the whole list in ONE API call — far cheaper and faster than
    # looping embed_query. (Some models also encode documents/queries asymmetrically.)
    batch_embedding = embeddings.embed_documents(text)
    for i, emb in enumerate(batch_embedding):
        print(f"Text {i+1} - Vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")


def similarity_search():

    # Documents

    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    # embed documents and query
    doc_vector = embeddings_model.embed_documents(docs)
    query_vector = embeddings_model.embed_query(query)

    # Cosine (angle) rather than Euclidean distance: it ignores vector magnitude, so
    # long and short texts about the same topic still score as similar. This is the
    # exact computation a vector store does for you under the hood.
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f"  {score:.4f}: {doc}")


# Caching ---
def embedding_caching():
    import tempfile

    from langchain_classic.embeddings.cache import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore

    with tempfile.TemporaryDirectory() as tempdir:
        store = LocalFileStore(root_path=tempdir)

        # Caches by hash of the text, so re-indexing an unchanged corpus costs nothing.
        # namespace should encode the embedding MODEL — otherwise switching models would
        # silently return stale vectors of the wrong dimensionality from the same keys.
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings_model,
            document_embedding_cache=store,
            namespace="exercise",
        )

        text = "What is Reinforcement Learning?"

        # Note: only embed_documents is cached — embed_query is not, by design, since
        # queries are usually unique and caching them would just bloat the store.
        # First call - hits API
        print("First call (API):")
        vectors1 = cached_embeddings.embed_documents([text])
        print(f"  Embedded {len(vectors1)} documents")

        # Second call - from cache
        print("\nSecond call (Cache):")
        vectors2 = cached_embeddings.embed_documents([text])
        print(f"  Embedded {len(vectors2)} documents")

        # Verify same results
        print(f"\nSame vectors: {np.allclose(vectors1[0], vectors2[0])}")


if __name__ == "__main__":
    # batch_embeddings()
    # basic_embeddings()
    # similarity_search()
    embedding_caching()
