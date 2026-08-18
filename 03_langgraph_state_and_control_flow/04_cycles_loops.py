import operator
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

"""
Cycles and Loops in LangGraph
Self-correcting agents and iterative refinement
"""

llm = init_chat_model("gpt-4o-mini", temperature=0.0)

GRAPH_DIR = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "graphs"
    / "03_langgraph_state_and_control_flow"
)


def log_step(title: str, **details: object) -> None:
    """Print a titled section with aligned key/value details.

    Keeps node functions free of ad-hoc print formatting — call once per
    node with whatever's worth showing, e.g. log_step("RESEARCH", depth="1/2", query=query).
    """
    print(f"\n{'─' * 50}\n{title}")
    for key, value in details.items():
        print(f"   {key}: {value}")


def visualize_graph(app, name: str) -> None:
    """Print the mermaid source and save a PNG render for a compiled graph.

    `name` should be the descriptive kebab-case PNG filename without its extension.
    """
    print("\n--- Mermaid Graph ---")
    print(app.get_graph().draw_mermaid())

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    png_path = GRAPH_DIR / f"{name}.png"
    # draw_mermaid_png() calls the remote mermaid.ink renderer, so it needs network access.
    png_path.write_bytes(app.get_graph().draw_mermaid_png())
    print(f"\nGraph saved to {png_path}")


class CodeGenState(TypedDict):
    task: str
    code: str
    # Accumulated (not overwritten) so the full error history survives every loop pass;
    # the generator node reads errors[-1] to fix the most recent failure.
    errors: Annotated[list[str], operator.add]
    iteration: int
    # Carried IN STATE rather than as a module constant so the cap travels with the run
    # and can be tuned per invocation.
    max_iterations: int
    success: bool


def demo_self_correcting_code():
    """Self-correcting code generator."""

    def generate_code(state: CodeGenState) -> dict:
        print("Generating code...")

        if state["iteration"] == 0:
            # First attempt
            prompt = f"Write Python code for: {state['task']}\nReturn only the code."
        else:
            # Correction attempt
            prompt = (
                f"Fix this Python code:\n{state['code']}\n\n"
                f"Errors:\n{state['errors'][-1]}\n\n"
                "Return only the corrected code."
            )

        response = llm.invoke(prompt)
        code = str(response.content).strip()

        # LLMs wrap code in ```python fences even when told not to; splitting on ```
        # takes the fenced body, then the language tag is stripped off the front.
        # Clean up markdown code blocks if present
        if code.startswith("```"):
            code = code.split("```")[1]
            code = code.removeprefix("python")

        return {"code": code, "iteration": state["iteration"] + 1}

    def validate_code(state: CodeGenState) -> dict:
        print("Validating...")

        code = state["code"]

        # Step 1: Does it compile?
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return {"errors": [f"SyntaxError: {e}"], "success": False}

        # Step 2: Does it RUN and produce correct results?
        test_cases = [
            ([3, 1, 4, 1, 5, 9], 5),  # normal case
            ([1, 1, 1], None),  # all same → no second largest
            ([7], None),  # single element
            ([3, -1, 3, 5, 5], 3),  # duplicates at top
        ]

        # exec into a throwaway namespace so the generated code's definitions can be
        # looked up by name afterwards without polluting this module's globals.
        # (Demo only — exec'ing model output is unsafe outside a sandbox.)
        namespace = {}
        try:
            exec(code, namespace)
        # Broad `except Exception` on purpose: ANY failure must become feedback text
        # for the next repair iteration rather than crashing the graph.
        except Exception as e:
            return {"errors": [f"Runtime error: {e}"], "success": False}

        if "solve" not in namespace:
            return {"errors": ["Function 'solve' not found in code"], "success": False}

        # Fail fast on the first mismatch: one concrete counter-example is a better
        # repair prompt than a wall of failures.
        for inputs, expected in test_cases:
            try:
                result = namespace["solve"](inputs)
                if result != expected:
                    return {
                        "errors": [
                            f"solve({inputs}) returned {result}, expected {expected}"
                        ],
                        "success": False,
                    }
            except Exception as e:
                return {"errors": [f"solve({inputs}) raised {e}"], "success": False}

        return {"success": True}

    def should_continue(state: CodeGenState) -> Literal["generate", "end"]:
        if state["success"] or state["iteration"] >= state["max_iterations"]:
            return "end"
        else:
            return "generate"

    # No-op terminal node. It exists purely to give the conditional edge a concrete
    # "end" target: routing straight to END is possible but a real node is easier to
    # extend (logging, persistence) and shows up in the rendered graph.
    def finalize(state: CodeGenState) -> dict:
        return dict(state)

    graph = StateGraph(CodeGenState)

    graph.add_node("generate", generate_code)
    graph.add_node("validate", validate_code)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate", should_continue, {"generate": "generate", "end": "finalize"}
    )  # Loop back to "generate" if not successful and under max iterations, otherwise go to "finalize"
    graph.add_edge("finalize", END)

    app = graph.compile()

    visualize_graph(app, "self-correcting-code-loop")

    print("Self-Correcting Code Generator:\n")

    result = app.invoke(
        {
            "task": "a function that calculates factorial recursively",
            "code": "",
            "errors": [],
            "iteration": 0,
            "max_iterations": 5,
            "success": False,
        }
    )

    print(f"Task: {result['task']}")
    print(f"Iterations: {result['iteration']}")
    print(f"Success: {result['success']}")
    print(f"Final Code:\n{result['code']}")


class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]
    # NOTE: no reducer here, so each node's return REPLACES the list rather than
    # extending it — only the newest question survives, unlike `findings` above.
    questions: list[str]
    iteration: int
    max_depth: int
    summary: str


def demo_iterative_research():
    """Iterative research that goes deeper based on findings."""

    def research(state: ResearchState) -> dict:
        if state["iteration"] == 0:
            query = f"Give me 3 key facts about: {state['topic']}"
            log_step(
                "📚 RESEARCH", depth=f"1/{state['max_depth']}", topic=state["topic"]
            )
        else:
            question = state["questions"][-1] if state["questions"] else "elaborate"
            query = f"Based on these findings:\n{state['findings'][-1]}\n\nGo deeper: {question}"
            log_step(
                "📚 RESEARCH",
                depth=f"{state['iteration'] + 1}/{state['max_depth']}",
                following_up_on=question,
            )

        response = llm.invoke(query)
        content = str(response.content)
        print(f"   ✅ {len(content.splitlines())} lines — {content[:120]}...")
        return {"findings": [content]}

    def generate_questions(state: ResearchState) -> dict:
        log_step("🤔 QUESTIONING", analyzing="latest findings")

        response = llm.invoke(
            f"Based on this finding:\n{state['findings'][-1]}\n\n"
            "What's one deeper question to explore? Reply with just the question."
        )

        content = str(response.content)
        print(f"   Next question: {content.strip()}")

        # The iteration counter is bumped HERE, not in research(), so one "depth" equals
        # a full research -> question cycle and the router below counts it correctly.
        return {"questions": [content], "iteration": state["iteration"] + 1}

    def synthesize(state: ResearchState) -> dict:
        log_step("🧬 SYNTHESIZE", rounds=len(state["findings"]))

        # Fan-in step: every accumulated round is folded into one final answer.
        all_findings = "\n\n".join(state["findings"])
        response = llm.invoke(
            f"Synthesize these findings into a coherent summary:\n\n{all_findings}"
        )

        content = str(response.content)
        print(f"   ✅ Summary generated ({len(content.split())} words)")
        return {"summary": content}

    def should_continue(state: ResearchState) -> Literal["research", "synthesize"]:
        done = state["iteration"] >= state["max_depth"]
        log_step(
            "🏁 ROUTER" if done else "🔄 ROUTER",
            depth=f"{state['iteration']}/{state['max_depth']}",
            decision="synthesize" if done else "go deeper",
        )
        return "synthesize" if done else "research"

    graph = StateGraph(ResearchState)

    graph.add_node("research", research)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "research")
    graph.add_edge("research", "generate_questions")
    graph.add_conditional_edges(
        "generate_questions",
        should_continue,
        {"research": "research", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    app = graph.compile()

    print("=" * 50)
    print("🔬 ITERATIVE RESEARCH WORKFLOW")
    print("=" * 50)

    result = app.invoke(
        {
            "topic": "quantum computing applications",
            "findings": [],
            "questions": [],
            "iteration": 0,
            "max_depth": 2,
            "summary": "",
        }
    )

    print(f"\n{'=' * 50}")
    print("📊 RESEARCH COMPLETE")
    print(f"   Topic: {result['topic']}")
    print(f"   Depth reached: {result['iteration']}")
    print(f"   Findings collected: {len(result['findings'])}")
    print(f"   Questions explored: {len(result['questions'])}")
    print(f"\n📝 Final Summary:\n{result['summary']}")


if __name__ == "__main__":
    # demo_self_correcting_code()
    demo_iterative_research()
