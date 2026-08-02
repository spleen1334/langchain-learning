"""
LangSmith Setup and Observability
Production monitoring for LangChain/LangGraph
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable

load_dotenv()

# Tracing is driven entirely by env vars (LANGSMITH_TRACING / LANGSMITH_API_KEY /
# LANGSMITH_PROJECT) — no code changes needed to instrument a chain. Set AFTER
# load_dotenv() so it overrides whatever the .env file said.
os.environ["LANGSMITH_TRACING"] = "true"


# LCEL chains are traced automatically; @traceable adds a PARENT span around them so
# your own function shows up as one logical run with the chain nested inside it.
@traceable(name="basic_chaining")
def demo_basic_tracing():
    """Basic LangSmith tracing."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")

    chain = prompt | llm | StrOutputParser()

    print("Basic Tracing Demo:\n")
    print("Running chain with LangSmith tracing enabled...")

    result = chain.invoke({"topic": "machine learning"})

    print(f"Result: {result}")
    print("\nCheck LangSmith dashboard for trace details.")


# Tags are the filter/grouping dimension in the LangSmith UI — the practical way to
# separate e.g. prod from dev traffic within one project.
@traceable(name="named_runs_demo", tags=["production", "summarization"])
def demo_named_runs():
    """Name your runs for easier identification."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("Summarize: {text}")

    chain = prompt | llm | StrOutputParser()

    print("\nNamed Runs Demo:\n")

    result = chain.invoke(
        {"text": "LangSmith provides observability for LLM applications."}
    )

    print(f"Result: {result}")
    print("Run tagged with 'production', 'summarization'")


@traceable(name="trace_with_metadata_demo", tags=["metadata", "filtering"])
def demo_trace_with_metadata(user_id: str, request_type: str):
    """Add metadata to traces for filtering."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # @traceable records the function's ARGUMENTS as the run's inputs, so user_id and
    # request_type land in the trace automatically — that's the "metadata" here.
    # Corollary: never pass secrets/PII to a traced function unless you want them stored.
    result = llm.invoke(f"Hello from user {user_id}")

    return result.content


if __name__ == "__main__":
    demo_basic_tracing()
    demo_named_runs()
    demo_trace_with_metadata(user_id="user_123", request_type="greeting")
