"""
Section 4 Project: Multi-Agent Research System
Combines all patterns from Section 4 into a working research pipeline.

Patterns used:
- Supervisor architecture (4.3)
- Parallel execution (4.5) via Send API
- Shared state / blackboard (4.6)
- Iterative refinement loop
"""

import json
import operator
from typing import Annotated, Literal, cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ============================================================
# State Schema
# ============================================================


class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    search_queries: list[str]
    # MUST have a reducer: several search_agent instances write this key CONCURRENTLY,
    # and without operator.add LangGraph raises an InvalidUpdateError on the conflict.
    # With it, every agent's list is merged into one.
    findings: Annotated[list[dict], operator.add]
    analysis: str
    report: str
    quality_score: float
    quality_feedback: str
    iteration: int


# State for individual search tasks (used with Send API)
# A SEPARATE, narrower schema for the Send-spawned workers: each one only needs its own
# query, not the whole research state. `findings` is repeated here because that's the key
# the worker writes back into the parent state.
class SearchTaskState(TypedDict):
    search_query: str
    findings: Annotated[list[dict], operator.add]


# ============================================================
# Node: Supervisor — Plans the research
# ============================================================


def supervisor(state: ResearchState) -> dict:
    """Plans research by generating targeted search queries."""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a research supervisor. Given a topic, generate exactly 3 "
                    "specific search queries that will cover different angles of the topic. "
                    "Return ONLY a JSON array of strings. No markdown formatting."
                )
            ),
            HumanMessage(content=f"Research topic: {state['topic']}"),
        ]
    )

    try:
        queries = json.loads(str(response.content))
    except json.JSONDecodeError:
        # Fallback: split by newlines
        queries = [
            f"{state['topic']} overview",
            f"{state['topic']} latest developments",
            f"{state['topic']} practical applications",
        ]

    return {
        # Hard cap at 3 regardless of what the model returned — the fan-out width below
        # is driven by this list, so an over-eager model would multiply the API spend.
        "search_queries": queries[:3],
        "messages": [
            AIMessage(
                content=f"[SUPERVISOR]: Planned {len(queries)} research queries: {queries}",
                name="supervisor",
            )
        ],
    }


# ============================================================
# Node: Search Agent — Executes a single search query
# (Launched in parallel via Send API)
# ============================================================


def search_agent(state: SearchTaskState) -> dict:
    """
    Executes one search query and returns findings.
    Each instance runs in parallel via the Send API.
    """
    query = state["search_query"]

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a web research agent. For the given search query, "
                    "provide 2-3 key findings. Each finding should have a 'title' "
                    "and 'detail' field. Return a JSON array. No markdown."
                )
            ),
            HumanMessage(content=f"Search query: {query}"),
        ]
    )

    try:
        results = json.loads(str(response.content))
    except json.JSONDecodeError:
        results = [{"title": query, "detail": response.content}]

    # Tagging is essential under fan-in: once operator.add merges every worker's results
    # into one flat list, this field is the only record of which query produced what.
    for r in results:
        r["source_query"] = query

    return {"findings": results}


# ============================================================
# Edge: Fan-out searches using Send API
# ============================================================


def dispatch_searches(state: ResearchState) -> list[Send]:
    """Dynamically create parallel search tasks using Send API."""
    # This is the real map-reduce fan-out (vs. the fixed START->3-nodes pattern):
    # Send(node_name, payload) spawns ONE INSTANCE of "search_agent" per query, each with
    # its own private state, all running concurrently. The number of instances is decided
    # at RUNTIME from the data, so the graph doesn't need N pre-declared nodes.
    # Each instance's return value is merged back into the parent via the reducers,
    # which is the "reduce" half — hence findings needing operator.add.
    return [
        Send("search_agent", {"search_query": query, "findings": []})
        for query in state["search_queries"]
    ]


# ============================================================
# Node: Analyst — Synthesizes all findings
# ============================================================


def analyst(state: ResearchState) -> dict:
    """Reads all findings from the blackboard and synthesizes."""
    findings_text = json.dumps(state["findings"], indent=2)

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a research analyst. Synthesize the collected findings into "
                    "a clear analysis. Identify:\n"
                    "1. Key themes across all findings\n"
                    "2. Any contradictions or gaps\n"
                    "3. The most important insights\n\n"
                    "Write 2-3 paragraphs."
                )
            ),
            HumanMessage(
                content=(
                    f"Research topic: {state['topic']}\n\n"
                    f"Collected findings:\n{findings_text}"
                )
            ),
        ]
    )

    return {
        "analysis": response.content,
        "messages": [
            AIMessage(content=f"[ANALYST]: {response.content}", name="analyst")
        ],
    }


# ============================================================
# Node: Report Writer — Produces the final report
# ============================================================


def report_writer(state: ResearchState) -> dict:
    """Writes a structured research report from the analysis."""

    # Include quality feedback if this is a revision
    # Same node serves both the first draft and every revision; the iteration counter is
    # what makes it inject the critic's feedback on later passes.
    revision_note = ""
    if state["iteration"] > 0 and state.get("quality_feedback"):
        revision_note = (
            f"\n\nIMPORTANT — This is revision #{state['iteration']}. "
            f"Address this feedback: {state['quality_feedback']}"
        )

    # The higher-temperature model is used HERE only: prose benefits from variation,
    # whereas planning/analysis/review stay at temperature=0 for consistency.
    response = creative_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a report writer. Produce a well-structured research report "
                    "with these sections:\n"
                    "1. Executive Summary (2-3 sentences)\n"
                    "2. Key Findings (bullet points)\n"
                    "3. Analysis (1-2 paragraphs)\n"
                    "4. Recommendations (3 actionable items)\n\n"
                    "Use markdown formatting. Be specific and actionable."
                    f"{revision_note}"
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n\n"
                    f"Analysis:\n{state['analysis']}\n\n"
                    f"Raw findings:\n{json.dumps(state['findings'][:6], indent=2)}"
                )
            ),
        ]
    )

    return {
        "report": response.content,
        "messages": [
            AIMessage(
                content=f"[REPORT WRITER]: Report {'revised' if state['iteration'] > 0 else 'drafted'}.",
                name="report_writer",
            )
        ],
    }


# ============================================================
# Node: Quality Checker — Reviews and scores the report
# ============================================================


class QualityReview(BaseModel):
    score: float = Field(description="Quality score from 0.0 to 1.0")
    feedback: str = Field(description="Specific feedback for improvement")
    approved: bool = Field(description="Whether the report meets quality standards")


def quality_checker(state: ResearchState) -> dict:
    """Reviews the report and either approves or sends back for revision."""

    review_llm = llm.with_structured_output(QualityReview)

    review = cast(
        QualityReview,
        review_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a quality reviewer. Score the report on:\n"
                        "- Completeness: Does it cover the topic well?\n"
                        "- Clarity: Is it well-written and easy to understand?\n"
                        "- Actionability: Are recommendations specific?\n\n"
                        "Score from 0.0 to 1.0. Approve if score >= 0.7.\n"
                        "If this is iteration 2 or higher, be more lenient."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {state['topic']}\n"
                        f"Iteration: {state['iteration']}\n\n"
                        f"Report:\n{state['report']}"
                    )
                ),
            ]
        ),
    )

    # Local variable only — used for the log message below. The ACTUAL loop exit is
    # decided by quality_gate(), which re-derives the same condition from state.
    # Force approve after 2 iterations to prevent infinite loops
    approved = review.approved or state["iteration"] >= 2

    return {
        "quality_score": review.score,
        "quality_feedback": review.feedback,
        "iteration": state["iteration"] + 1,
        "messages": [
            AIMessage(
                content=(
                    f"[QUALITY CHECK]: Score {review.score:.1f} — "
                    f"{'APPROVED' if approved else 'REVISION NEEDED'}: {review.feedback}"
                ),
                name="quality_checker",
            )
        ],
    }


# ============================================================
# Build the full research system
# ============================================================


def create_research_system():
    """
    Builds the multi-agent research system graph.

    Flow:
    1. Supervisor plans search queries
    2. Search agents run in parallel (Send API)
    3. Analyst synthesizes findings
    4. Report writer produces report
    5. Quality checker reviews — loops back if needed
    """

    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("search_agent", search_agent)
    graph.add_node("analyst", analyst)
    graph.add_node("report_writer", report_writer)
    graph.add_node("quality_checker", quality_checker)

    # Edges
    graph.add_edge(START, "supervisor")

    # Send-based dispatch is wired as a CONDITIONAL edge whose function returns Send
    # objects instead of a route key; the third arg just declares the possible targets
    # so the graph can be drawn correctly.
    graph.add_conditional_edges("supervisor", dispatch_searches, ["search_agent"])

    # One edge covers all spawned instances: LangGraph waits for EVERY Send task to
    # finish before running analyst once with the merged findings.
    graph.add_edge("search_agent", "analyst")

    # Analyst → report writer
    graph.add_edge("analyst", "report_writer")

    # Report writer → quality checker
    graph.add_edge("report_writer", "quality_checker")

    # Quality gate: approve or revise
    graph.add_conditional_edges(
        "quality_checker", quality_gate, {"report_writer": "report_writer", "end": END}
    )

    return graph.compile()


# Demos
def demo_research_with_streaming():
    """Run the research system with step-by-step streaming output."""

    system = create_research_system()
    graph = system.get_graph()
    png_data = graph.draw_mermaid_png()

    with open("research_graph.png", "wb") as f:
        f.write(png_data)

    topic = "Best practices for building multi-agent AI systems"
    print(f"Streaming Research: {topic}\n")

    initial_state: ResearchState = {
        "messages": [],
        "topic": topic,
        "search_queries": [],
        "findings": [],
        "analysis": "",
        "report": "",
        "quality_score": 0.0,
        "quality_feedback": "",
        "iteration": 0,
    }

    # stream_mode="updates" yields only each node's DELTA as it completes (vs "values",
    # which re-emits the entire state every step) — ideal for a progress log.
    for step in system.stream(initial_state, stream_mode="updates"):
        for node_name, update in step.items():
            print(f"[{node_name}] completed")

            # Show interesting state changes
            if update.get("search_queries"):
                print(f"  Planned queries: {update['search_queries']}")
            if update.get("findings"):
                print(f"  Found {len(update['findings'])} results")
            if "quality_score" in update:
                print(f"  Quality score: {update['quality_score']:.1f}")
            if update.get("report"):
                print(f"  Report length: {len(update['report'])} chars")

        print()


# ============================================================
# Routing: Quality gate
# ============================================================


def quality_gate(state: ResearchState) -> Literal["report_writer", "end"]:
    """Route back to writer if quality is insufficient."""
    # Two exits: good enough, OR budget spent. The iteration cap is the hard guarantee —
    # without it a persistently harsh reviewer would loop until the recursion limit.
    # Note iteration was already incremented by quality_checker before this runs.
    if state["quality_score"] >= 0.7 or state["iteration"] >= 2:
        return "end"
    return "report_writer"


def demo_individual_search():
    """Demo just the search agent for testing."""

    print("Individual Search Agent Test:\n")

    # Nodes are plain functions of state, so they can be called directly with a dict —
    # no graph, no API of its own. That makes each agent independently unit-testable.
    result = search_agent(
        {"search_query": "LangGraph multi-agent patterns", "findings": []}
    )

    print("Findings from search:")
    for f in result["findings"]:
        print(f"  - {f.get('title', 'N/A')}: {f.get('detail', 'N/A')[:80]}...")


def demo_full_research():
    """Run the complete research pipeline."""

    system = create_research_system()

    print("Multi-Agent Research System Demo")
    print("=" * 60)

    topic = "The impact of AI agents on software development in 2026"
    print(f"Topic: {topic}\n")

    result = system.invoke(
        {
            "messages": [],
            "topic": topic,
            "search_queries": [],
            "findings": [],
            "analysis": "",
            "report": "",
            "quality_score": 0.0,
            "quality_feedback": "",
            "iteration": 0,
        }
    )

    # Print the conversation trace
    print("Agent Activity Log:")
    print("-" * 40)
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"  {msg.content[:120]}...")
    print()

    # Print final stats
    print(f"Total findings collected: {len(result['findings'])}")
    print(f"Quality score: {result['quality_score']:.1f}")
    print(f"Iterations: {result['iteration']}")
    print()

    # Print the final report
    print("=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)
    print(result["report"])


if __name__ == "__main__":
    # Run the individual search agent demo
    # demo_individual_search()
    # print("\n" + "=" * 50 + "\n")
    # Run the full research system demo with streaming output
    # demo_research_with_streaming()
    # print("\n" + "=" * 50 + "\n")
    # Run the full research system demo without streaming output
    demo_full_research()
