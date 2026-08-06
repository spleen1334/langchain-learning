import operator
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()


class ConversationState(TypedDict):
    # Plain strings here (not Message objects), so operator.add is enough — no need for
    # the add_messages reducer, which exists to dedupe/coerce real BaseMessage objects.
    messages: Annotated[list, operator.add]
    # No reducer => last write wins; each run overwrites the classification.
    sentiment: str
    # Also unreduced, which is why the counter below never actually accumulates
    # across invoke() calls (see demo_conversation).
    response_count: int


def create_conversation_graph():
    llm = init_chat_model("gpt-4o-mini", temperature=0.7)

    # Define node function
    def analyze_sentiment(state: ConversationState) -> dict:
        """Analyze the sentiment of the last message."""
        last_message = state["messages"][-1]

        response = llm.invoke(
            [
                SystemMessage(
                    content="Classify sentiment as: positive, negative, or neutral. Reply with just the word."
                ),
                HumanMessage(content=last_message),
            ]
        )

        return {"sentiment": str(response.content).lower().strip()}

    def generate_response(state: ConversationState) -> dict:
        """Generate appropriate response based on sentiment."""
        sentiment = state["sentiment"]
        last_message = state["messages"][-1]

        system_prompts = {
            "positive": "Respond enthusiastically and build on their positive energy.",
            "negative": "Respond empathetically and offer support.",
            "neutral": "Respond helpfully and informatively.",
        }

        # .get with a default guards against the LLM returning anything off-script
        # (e.g. "Positive." or a full sentence) — free-text classification is not reliable.
        prompt = system_prompts.get(sentiment, system_prompts["neutral"])

        response = llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=last_message)]
        )

        return {"messages": [f"AI: {response.content}"], "response_count": 1}

    # Create graph
    graph = StateGraph(ConversationState)

    # Add nodes
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("generate_response", generate_response)

    # Add edges
    graph.add_edge(START, "analyze_sentiment")
    graph.add_edge("analyze_sentiment", "generate_response")
    graph.add_edge("generate_response", END)

    app = graph.compile()

    return app


def demo_conversation():
    app = create_conversation_graph()

    # Simulate a conversation

    test_messages = [
        "I just got promoted at work! I'm so excited!",
        "My computer crashed and I lost all my work...",
        "What's the weather like today?",
    ]

    print("Conversation Graph Demo:\n")

    # Each invoke() starts from a fresh state — without a checkpointer the graph keeps
    # nothing between runs, so these three messages are independent, not one conversation.
    for msg in test_messages:
        result = app.invoke(
            {"messages": [f"Human: {msg}"], "sentiment": "", "response_count": 0}
        )

        print(f"Input: {msg}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Response: {result['messages'][-1]}")
        print("-" * 40)


if __name__ == "__main__":
    demo_conversation()
