"""
Supervisor Architecture in LangGraph
One agent coordinates multiple specialist agents
"""

from typing import Annotated, Literal, cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    task_complete: bool
    final_response: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def create_supervisor_system():
    """Create a supervisor with specialist agents."""

    # Define the routing schema
    class RouteDecision(BaseModel):
        # Literal becomes an ENUM in the JSON schema sent to the provider, so the model
        # is constrained to these exact values. This is what makes routing safe: free-text
        # routing could return "the writer" and blow up the conditional-edge lookup.
        next: Literal["researcher", "writer", "critic", "FINISH"] = Field(
            description="The next agent to call, or FINISH if task is complete"
        )
        # Forcing an explicit rationale improves the routing decision itself (the model
        # must justify it) and gives you a readable trace.
        reasoning: str = Field(description="Why this agent was chosen")

    supervisor_llm = llm.with_structured_output(RouteDecision)

    # Supervisor node
    def supervisor(state: SupervisorState) -> dict:
        system_prompt = """You are a supervisor managing a team of specialists:

        1. researcher - Gathers information and facts
        2. writer - Creates content and text
        3. critic - Reviews and improves work

        Based on the conversation, decide which agent should act next.
        If the task is complete, respond with FINISH.

        Current conversation shows the progress so far."""

        # The supervisor routes off the FULL shared message history — that transcript is
        # its only view of what the specialists have already produced.
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        decision = cast(RouteDecision, supervisor_llm.invoke(messages))

        # On FINISH, deliberately no message is appended — nothing to add to the
        # transcript, and task_complete is what the router below actually checks.
        if decision.next == "FINISH":
            return {"next_agent": "FINISH", "task_complete": True}

        return {
            "next_agent": decision.next,
            "messages": [
                # The "[Supervisor]"/"[Writer]"/... prefixes are how agents identify each
                # other's contributions in a single shared message list — LangGraph has no
                # per-agent channel here, so the tag IS the sender identity.
                AIMessage(
                    content=f"[Supervisor] Routing to {decision.next}: {decision.reasoning}"
                )
            ],
        }

    # Define specialist agents (for demo purposes, they just echo the task)
    def researcher(state: SupervisorState) -> dict:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a research specialist. Gather facts and information relevant to the task. Be thorough but concise.",
                ),
                (
                    "human",
                    "Task context:\n{context}\n\nProvide your research findings.",
                ),
            ]
        )

        # Pull the ORIGINAL user request rather than the last message, so the researcher
        # works from the real task instead of the supervisor's routing chatter.
        task = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )

        response = llm.invoke(prompt.format_messages(context=task))

        return {"messages": [AIMessage(content=f"[Researcher] {response.content}")]}

    def writer(state: SupervisorState) -> dict:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a writing specialist. Create clear, engaging content based on the available information.",
                ),
                ("human", "Previous work:\n{context}\n\nWrite the content."),
            ]
        )

        # Sliding window of the last few messages: keeps the specialist's prompt bounded
        # as the shared transcript grows (the critic below uses an even tighter window).
        context = "\n".join([str(m.content) for m in state["messages"][-5:]])
        response = llm.invoke(prompt.format_messages(context=context))

        return {"messages": [AIMessage(content=f"[Writer] {response.content}")]}

    def critic(state: SupervisorState) -> dict:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a quality critic. Review the work and provide constructive feedback. If the work is good, say so.",
                ),
                ("human", "Work to review:\n{context}\n\nProvide your critique."),
            ]
        )

        context = "\n".join([str(m.content) for m in state["messages"][-3:]])
        response = llm.invoke(prompt.format_messages(context=context))

        return {"messages": [AIMessage(content=f"[Critic] {response.content}")]}

    def finalize(state: SupervisorState) -> dict:
        # Scans BACKWARDS for the newest [Writer] output — the deliverable is the writer's
        # latest draft, not the critic's feedback or the supervisor's routing note.
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and "[Writer]" in str(msg.content):
                content = str(msg.content).replace("[Writer] ", "")
                return {"final_response": content}

        return {"final_response": "Task completed."}

    # Returns the agent NAME straight from state; since the Literal above guarantees the
    # value is one of the mapped keys, no validation/fallback is needed here.
    def route_to_agent(state: SupervisorState) -> str:
        if state.get("task_complete"):
            return "finalize"
        return state["next_agent"]

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "writer": "writer",
            "critic": "critic",
            "finalize": "finalize",
        },
    )

    # Star topology: specialists never call each other, they always return to the
    # supervisor, which re-decides. That's what distinguishes supervisor architecture
    # from a fixed pipeline — and it's why the graph can loop indefinitely if the
    # supervisor never says FINISH (LangGraph's recursion limit is the only backstop).
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("finalize", END)

    return graph.compile()


def demo_supervisor():
    """Demo the supervisor system."""

    agent = create_supervisor_system()

    print("Supervisor Agent Demo:\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Write a short blog post about the benefits of AI in healthcare"
                )
            ],
            "next_agent": "",
            "task_complete": False,
            "final_response": "",
        }
    )

    print("Agent conversation:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")

    print(f"\n\nFinal Response:\n{result['final_response']}")


def demo_supervisor_trace():
    """Show supervisor decision-making."""

    agent = create_supervisor_system()

    print("\nSupervisor Decision Trace:\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Create a marketing tagline for a new coffee brand"
                )
            ],
            "next_agent": "",
            "task_complete": False,
            "final_response": "",
        }
    )

    print("Routing decisions:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and "[Supervisor]" in msg.content:
            print(f"  → {msg.content}")


if __name__ == "__main__":
    # demo_supervisor()
    demo_supervisor_trace()
