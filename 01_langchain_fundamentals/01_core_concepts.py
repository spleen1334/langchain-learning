"""
LangChain Core Concepts - LCEL and Runnables
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables."""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    # Every component is a Runnable, so `|` builds a single composed Runnable.
    # StrOutputParser at the end unwraps the AIMessage into a plain str.
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question": "What is LangChain?"})
    print(f"Response: {result}")

    return chain


def demo_batch_exectution():
    """Demonstrate batch execution for multiple inputs."""
    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Batch - run with multiple inputs
    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "Where is the nearest restaurant?"},
    ]
    # .batch() fires the requests concurrently (not a sequential loop) and preserves input order.
    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")


def demo_streaming():
    """Demonstrate streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    # .stream() yields token chunks as they arrive instead of blocking until the full
    # answer is ready like .invoke(). Streaming only works if every component in the
    # chain supports it — a non-streaming step would buffer the whole output first.
    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming


def demo_schema_inspection():
    """Demonstrate input/output schema inspection."""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Schemas are inferred from the chain's ends: input comes from the prompt's
    # template variables, output from the last component (str for StrOutputParser).
    # This is what powers automatic validation and the LangServe API docs.
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")


# ------- Exercise the demos -------#
# Exercise: Build your first chain
def exercise_first_chain():
    """
    EXERCISE: Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Returns just the tagline as a string

    Test with: product="AI Course", audience="developers"
    """

    # YOUR CODE HERE
    prompt = ChatPromptTemplate.from_template(
        "Create a marketing tagline for a product named '{product}' targeting '{audience}'."
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Test the chain
    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Marketing Tagline: {result}")


def new_way():
    # init_chat_model infers the provider from the model name and returns the right
    # class, so you can swap providers via config/env without changing imports.
    model = init_chat_model("gpt-4o-mini", temperature=0.7, max_tokens=1500)

    # Or provider-specific (still works)

    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    openai_model = ChatOpenAI(
        # max_retries adds built-in exponential backoff on rate limits / transient 5xx;
        # timeout caps a single request so a hung call can't stall the whole chain.
        model="gpt-4o-mini", temperature=0.7, max_tokens=1500, timeout=30, max_retries=3
    )

    anthropic_model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

    return model, openai_model, anthropic_model


if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_exectution()
    # demo_streaming()
    # demo_schema_inspection()
    exercise_first_chain()
