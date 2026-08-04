# Model                     Dimensions  Cost per 1M tokens  Best For
# text-embedding-3-small    1536        $0.02                General use
# text-embedding-3-large    3072        $0.13                High accuracy
# text-embedding-ada-002    1536        $0.10                Legacy

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def demo_huggingface_embeddings():
    """Runs locally (no API cost, no data leaves the machine). Fewer dimensions
    = smaller index and faster search, at some accuracy cost vs the OpenAI
    models above. Dimensions are NOT interchangeable: switching embedding
    models means re-indexing everything, since old vectors can't be compared
    against new ones."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )  # 384 dimensions
    return embeddings


def demo_ollama_embeddings():
    """Ollama: local embeddings served through an Ollama model."""
    embeddings = OllamaEmbeddings(model="llama2-7b-embedding-q4_0")
    return embeddings


def demo_openai_embed_query():
    """Embed a single piece of text and inspect the resulting vector."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    text = "This is a sample text to be embedded."
    embedding = embeddings.embed_query(text)
    print(len(embedding))  # Should print 1536 for text-embedding-3-small


def demo_openai_embed_documents():
    """Embed multiple texts at once via embed_documents()."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    embeds = embeddings.embed_documents(
        ["This is the first document.", "This is the second document."]
    )
    print(f"Number of embeddings returned: {len(embeds)}")  # Should print 2
    print(f"Length of each embedding: {len(embeds[0])}")  # Should print 1536


if __name__ == "__main__":
    demo_openai_embed_query()
    demo_openai_embed_documents()
