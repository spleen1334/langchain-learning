"""
Section 1 Project: Smart Q&A Bot
A production-ready question-answering bot with structured output
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

load_dotenv()

# -- LangSmith Configuration --
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    # setdefault, so an explicit LANGSMITH_PROJECT in .env still wins.
    os.environ.setdefault("LANGSMITH_PROJECT", "Smart Q&A Bot Project")
    print(f"LangSmith is configured. - Project: {os.getenv('LANGSMITH_PROJECT')}")


# Schema Definition
class QAResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    reasoning: str = Field(description="The reasoning behind the answer provided.")
    # default_factory (not default=[]) is the safe way to default a mutable field —
    # it builds a fresh list per instance instead of sharing one across all of them.
    follow_up_questions: list[str] = Field(
        description="A list of follow-up questions related to the topic.",
        default_factory=list,
    )
    sources_needed: bool = Field(
        description="Indicates whether sources are needed for the answer.",
        default=False,
    )

    # Bot implementation


class SmartQABot:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        # Low but non-zero: enough variation for natural follow-up suggestions while
        # keeping factual answers stable.
        temperature: float = 0.3,
    ):
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        ).with_structured_output(QAResponse)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a knowledgeable Q&A assistant.

Your guidelines:
- Answer questions accurately and concisely
- Be honest about uncertainty - set confidence to 'low' if unsure
- Provide clear reasoning for your answers
- Suggest relevant follow-up questions
- Indicate if external sources would help

Always respond with accurate, helpful information.""",
                ),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.model

    # run_type="chain" controls the icon/grouping in the LangSmith trace tree
    # ("llm", "tool", "retriever" are the other common values).
    @traceable(name="ask_question", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return response
        # Broad catch by design: callers are typed to always receive a QAResponse, so an
        # API outage or a schema-validation failure becomes a low-confidence answer
        # rather than an exception leaking into the UI.
        except Exception as e:
            # return a greaceful error response
            return QAResponse(
                answer="I'm sorry, I couldn't process your question at this time.",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=["Could you please try again later?"],
                sources_needed=True,
            )

    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: list[str]) -> list[QAResponse]:
        """Ask multiple questions in parallel."""
        inputs = [{"question": q} for q in questions]
        # Note this bypasses ask()'s error handling: .batch() raises if any item fails.
        # Use .batch(..., return_exceptions=True) to get per-item failures instead.
        return self.chain.batch(inputs)


# Demo Usage
def demo_qa_bot():
    bot = SmartQABot()

    questions = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "How does photosynthesis work?",
    ]

    print("=" * 60)
    print("SMART Q&A BOT DEMO")
    print("=" * 60)

    for question in questions:

        print(f"\n Question: {question}")
        print("-" * 40)

        response = bot.ask(question)

        print(f"Question: {question}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Follow-up Questions: {response.follow_up_questions}")
        print(f"Sources Needed: {response.sources_needed}")
        print("-" * 60)


@traceable(name="error_handling_demo", run_type="chain")
def demo_error_handling():
    """Demonstrate error handling."""

    bot = SmartQABot()

    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMO")
    print("=" * 60)

    # Test with a very long question (edge case)
    long_question = "What is " + "very " * 100 + "important?"

    response = bot.ask(long_question)
    print(f"Handled gracefully: {response.confidence}")


@traceable(name="batch_demo", run_type="chain")
def demo_batch_processing():
    """Demonstrate batch processing."""

    bot = SmartQABot()

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMO")
    print("=" * 60)

    responses = bot.ask_batch(questions)

    for q, r in zip(questions, responses):
        print(f"\n{q}")
        print(f"  -> {r.answer[:100]}...")
        print(f"  Confidence: {r.confidence}")


if __name__ == "__main__":

    try:
        demo_qa_bot()
        demo_batch_processing()
        demo_error_handling()

        print("\n" + "=" * 60)
        print("Section 1 Complete!")
        print("=" * 60)
        print(
            """
What you learned:
- LangChain ecosystem overview
- Environment setup with uv
- Core concepts: Runnables, LCEL, pipe operator
- Working with multiple LLM providers
- Prompt templates and message types
- Output parsers and structured output
- Building a production Q&A bot
- LangSmith tracing with @traceable decorator

Next: Section 2 - Chains, RAG & Memory
        """
        )
    finally:
        pass
    # uncomment the line below to flush traces to LangSmith, but you'll alse see an error at the end of a run, which is not harmful at all, but annoying!
    # Client().flush()  # Ensure all traces are sent to LangSmith
