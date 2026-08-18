"""
Agent Handoffs in LangGraph
Passing control and context between agents
"""

from typing import Annotated, Literal, cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class HandoffState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_reason: str
    context_summary: str


class HandoffDecision(BaseModel):
    handoff_to: Literal["sales", "support", "billing", "stay", "end"] = Field(
        description="Which agent to hand off to"
    )
    reason: str = Field(description="Reason for handoff")
    # The heart of a handoff: the triage agent writes a distilled briefing that the
    # receiving agent gets in its system prompt. Without it the specialist would have to
    # re-derive the situation from raw history — the "warm transfer" vs "cold transfer".
    context: str = Field(description="Key context to pass to next agent")


def create_customer_service_system():

    def triage_agent(state: HandoffState) -> dict:
        """Initial triage to route customer."""

        system = """You are a customer service triage agent. Your job is to:
        1. Understand the customer's need
        2. Route to the appropriate specialist:
           - sales: Product questions, purchases, upgrades
           - support: Technical issues, bugs, how-to questions
           - billing: Payments, invoices, refunds
           - end: Simple questions you can answer directly

        Analyze the customer's message and decide where to route them."""

        handoff_llm = llm.with_structured_output(HandoffDecision)

        messages = [SystemMessage(content=system)] + state["messages"]
        decision = cast(HandoffDecision, handoff_llm.invoke(messages))

        # "end" = triage handles it itself. Avoids a pointless specialist hop (and its
        # extra LLM call) for trivial questions.
        if decision.handoff_to == "end":
            # Answer directly
            response = llm.invoke(
                [
                    SystemMessage(
                        content="Provide a brief, helpful response to the customer."
                    ),
                    # unpack the messages to flatten it
                    *state["messages"],
                ]
            )
            return {
                "messages": [AIMessage(content=f"[Triage] {response.content}")],
                "current_agent": "end",
            }

        return {
            "current_agent": decision.handoff_to,
            "handoff_reason": decision.reason,
            "context_summary": decision.context,
            "messages": [
                AIMessage(
                    content=f"[Triage] Transferring to {decision.handoff_to}: {decision.reason}"
                )
            ],
        }

    def sales_agent(state: HandoffState) -> dict:
        """Sales specialist."""

        # Context is injected into the SYSTEM prompt (not as a user turn) so the
        # specialist treats it as briefing material rather than customer input.
        system = f"""You are a sales specialist. Context from triage: {state.get("context_summary", "None")}

            Help the customer with product questions and purchases.
            Be helpful and informative, not pushy."""

        response = llm.invoke([SystemMessage(content=system), *state["messages"]])

        return {
            "messages": [AIMessage(content=f"[Sales] {response.content}")],
            # "..._complete" marks the conversation as handled; each specialist edges
            # straight to END, so this is a status marker for callers, not a route key.
            "current_agent": "sales_complete",
        }

    def support_agent(state: HandoffState) -> dict:
        """Technical support specialist."""

        system = f"""You are a technical support specialist. Context from triage: {state.get("context_summary", "None")}

        Help the customer with technical issues.
        Be patient and provide step-by-step guidance."""

        response = llm.invoke([SystemMessage(content=system), *state["messages"]])

        return {
            "messages": [AIMessage(content=f"[Support] {response.content}")],
            "current_agent": "support_complete",
        }

    def billing_agent(state: HandoffState) -> dict:
        """Billing specialist."""

        system = f"""You are a billing specialist. Context from triage: {state.get("context_summary", "None")}

        Help the customer with billing questions.
        Be clear about policies and next steps."""

        response = llm.invoke([SystemMessage(content=system), *state["messages"]])

        return {
            "messages": [AIMessage(content=f"[Billing] {response.content}")],
            "current_agent": "billing_complete",
        }

    # Whitelist rather than a direct passthrough: anything unexpected (including the
    # schema's unused "stay" value) falls through to "end" instead of a missing-node error.
    def route_from_triage(state: HandoffState) -> str:
        agent = state["current_agent"]
        if agent in ["sales", "support", "billing"]:
            return agent
        return "end"

    graph = StateGraph(HandoffState)

    graph.add_node("triage", triage_agent)
    graph.add_node("sales", sales_agent)
    graph.add_node("support", support_agent)
    graph.add_node("billing", billing_agent)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        route_from_triage,
        {"sales": "sales", "support": "support", "billing": "billing", "end": END},
    )

    # One-way handoff: unlike the supervisor pattern, control never returns to triage.
    graph.add_edge("sales", END)
    graph.add_edge("support", END)
    graph.add_edge("billing", END)

    return graph.compile()


def demo_handoffs():
    """Demo customer service handoffs."""

    agent = create_customer_service_system()

    print("Customer Service Handoff Demo:\n")

    queries = [
        "My app keeps crashing when I try to upload photos",
        "I want to upgrade to the premium plan",
        "I was charged twice for my subscription",
        "What time do you close?",
    ]

    for query in queries:
        print(f"Customer: {query}")

        result = agent.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "current_agent": "",
                "handoff_reason": "",
                "context_summary": "",
            }
        )

        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                print(f"  {msg.content[:150]}...")

        print("-" * 50)


if __name__ == "__main__":
    demo_handoffs()
