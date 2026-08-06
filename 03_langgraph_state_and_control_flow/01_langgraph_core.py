"""
LangGraph Core Concepts
StateGraph, nodes, edges, and basic patterns
"""

import operator
from pathlib import Path
from typing import Annotated, cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

GRAPH_DIR = Path(__file__).parent / "graph"


def visualize_graph(app, name: str) -> None:
    """Print the mermaid source and save a PNG render for a compiled graph.

    `name` should identify the demo the graph belongs to (e.g. "demo_simple_graph");
    the PNG is written to graph/graph_<name>.png.
    """
    print("\n--- Mermaid Graph ---")
    print(app.get_graph().draw_mermaid())

    GRAPH_DIR.mkdir(exist_ok=True)
    png_path = GRAPH_DIR / f"graph_{name}.png"
    # draw_mermaid_png() calls the remote mermaid.ink renderer, so it needs network access.
    png_path.write_bytes(app.get_graph().draw_mermaid_png())
    print(f"\nGraph saved to {png_path}")


# Basic state
class SimpleState(TypedDict):
    input: str
    output: str
    step: int


def demo_simple_graph():

    # define node functions
    # A node returns a PARTIAL state dict; LangGraph merges it into the running state.
    # Keys you omit (here: "input") are left untouched, not cleared.
    def process(state: SimpleState) -> dict:
        # simple processing logic, for demo purposes
        return {"output": state["input"].upper(), "step": state["step"] + 1}

    # create graph
    graph = StateGraph(SimpleState)

    # add nodes
    graph.add_node("process", process)
    # add edges
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    # compile() validates the topology (unreachable nodes, missing START/END path) and
    # returns a Runnable — the graph is not executable until this happens.
    app = graph.compile()

    visualize_graph(app, "demo_simple_graph")

    # run app
    result = app.invoke({"input": "hello", "output": "", "step": 0})

    print("simple graph result:", result)
    print(
        f" Input: {result['input']}, Output: {result['output']}, Step: {result['step']}"
    )


# === State with Reducers ===


# The second arg of Annotated[...] is the REDUCER: how a node's returned value is
# combined with the existing value. Without a reducer the default is "overwrite",
# so each node would clobber the previous one's list instead of appending to it.
class AccumulatingState(TypedDict):
    messages: Annotated[list[str], operator.add]  # lists concatenate when merged
    count: Annotated[int, operator.add]  # counts sum when merged


def demo_accumulating_state():
    # Each node returns only its OWN delta (a 1-element list, count=1); operator.add
    # does the accumulating. Returning the full accumulated list here would double it.
    def step_one(state: AccumulatingState) -> dict:
        return {"messages": ["Step 1 executed"], "count": 1}

    def step_two(state: AccumulatingState) -> dict:
        return {"messages": ["Step 2 executed"], "count": 1}

    graph = StateGraph(AccumulatingState)

    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)

    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", END)

    app = graph.compile()

    visualize_graph(app, "demo_accumulating_state")

    result = app.invoke({"messages": ["Initial message"], "count": 0})

    print("\nAccumulating State Result:")
    print(f"  Messages: {result['messages']}")
    print(f"  Count: {result['count']}")


# === Message State (Common Pattern) ===

from langgraph.graph import add_messages


# add_messages is a smarter reducer than operator.add: it appends new messages, but it
# also coerces raw dicts/strings into Message objects, assigns ids, and UPSERTS by id —
# re-emitting a message with an existing id replaces it instead of duplicating it.
# That id-based replacement is what makes editing/removing history possible.
class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def demo_message_state():
    llm = init_chat_model("gpt-4o-mini", temperature=0)

    def chat_node(state: MessageState) -> dict:
        # The whole history is passed to the LLM; only the new reply is returned,
        # because add_messages appends it to what's already in state.
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessageState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    app = graph.compile()

    result = app.invoke({"messages": [HumanMessage(content="Say Hello in Tagalog")]})

    print("\nMessage State Result:")
    for msg in result["messages"]:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content}")


# === Multi-Node Graph ===
class MultiStepState(TypedDict):
    input: str
    analyzed: str
    enhanced: str
    final: str


def demo_multi_node_graph():
    llm = init_chat_model("gpt-4o-mini", temperature=0)

    def analyze_node(state: MultiStepState) -> dict:
        response = llm.invoke(
            [
                HumanMessage(
                    content=f"Analyze the following input and summarize it in one sentence: {state['input']}"
                )
            ]
        )
        return {"analyzed": response.content}

    def enhance(state: MultiStepState) -> dict:
        response = llm.invoke(
            [
                HumanMessage(
                    content=f"Take the following analysis and enhance it with more details: {state['analyzed']}"
                )
            ]
        )

        return {"enhanced": response.content}

    def finalize(state: MultiStepState) -> dict:
        response = llm.invoke(
            [
                HumanMessage(
                    content=f"Take the following enhanced analysis and finalize it into a concise summary: {state['enhanced']}"
                )
            ]
        )
        return {"final": response.content}

    graph = StateGraph(MultiStepState)
    graph.add_node("analyze_node", analyze_node)
    graph.add_node("enhance_node", enhance)
    graph.add_node("finalize_node", finalize)

    graph.add_edge(START, "analyze_node")
    graph.add_edge("analyze_node", "enhance_node")
    graph.add_edge("enhance_node", "finalize_node")
    graph.add_edge("finalize_node", END)

    app = graph.compile()

    visualize_graph(app, "demo_multi_node_graph")

    # Only "input" is supplied: TypedDict keys are not required at runtime, and each
    # node fills in its own key as the chain of edges progresses. The cast tells the
    # type checker what pyright can't infer from a partial dict literal.
    result = app.invoke(cast(MultiStepState, {"input": "Artificial intelligence"}))

    print("\nMulti-Node Graph Result:")
    print(f"  Input: {result['input']}")
    print(f"  Analyzed: {result['analyzed'][:100]}...")
    print(f"  Enhanced: {result['enhanced'][:100]}...")
    print(f"  Final: {result['final']}")


# Exercise
def exercise_first_langgraph():
    """
    EXERCISE: Create a LangGraph that:
    1. Takes a topic as input
    2. Node 1: Generates 3 questions about the topic
    3. Node 2: Answers one of the questions
    4. Returns both questions and answer
    """

    class QAState(TypedDict):
        topic: str
        questions: str
        answer: str

    llm = init_chat_model("gpt-4o-mini", temperature=0)

    def generate_questions(state: QAState) -> dict:
        response = llm.invoke(
            f"Generate 3 interesting questions about: {state['topic']}\n"
            "Format: numbered list"
        )
        return {"questions": response.content}

    def answer_question(state: QAState) -> dict:
        response = llm.invoke(
            f"Answer the first question from this list:\n{state['questions']}"
        )
        return {"answer": response.content}

    graph = StateGraph(QAState)

    graph.add_node("generate_questions", generate_questions)
    graph.add_node("answer_question", answer_question)

    graph.add_edge(START, "generate_questions")
    graph.add_edge("generate_questions", "answer_question")
    graph.add_edge("answer_question", END)

    app = graph.compile()

    result = app.invoke(cast(QAState, {"topic": "The future of renewable energy"}))

    print("\nExercise Result:")
    print(f"  Topic: {result['topic']}")
    print(f"  Questions: {result['questions']}")
    print(f"  Answer: {result['answer']}")


if __name__ == "__main__":
    demo_simple_graph()
    demo_accumulating_state()

    # go to .env LANGSMITH_TRACING=false to disable langsmith tracing for the next example, or set it to true to see the tracing in action
    demo_message_state()

    demo_multi_node_graph()
    exercise_first_langgraph()
