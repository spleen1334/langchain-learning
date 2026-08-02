"""
Human-in-the-Loop Patterns in LangGraph
Interrupt, review, modify, and resume
"""

from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─── Helper for visual separation ───
def phase_banner(phase_num: int, title: str):
    print(f"\n{'=' * 55}")
    print(f"  PHASE {phase_num}: {title}")
    print(f"{'=' * 55}")


def step_print(icon: str, label: str, detail: str = ""):
    print(f"\n{icon} [{label}] {detail}")


# ════════════════════════════════════════════════════════
# DEMO 1: Interrupt for Approval
# ════════════════════════════════════════════════════════


class ApprovalState(TypedDict):
    request: str
    draft: str
    approved: bool
    feedback: str
    final: str


def demo_interrupt_for_approval():
    """Interrupt execution for human approval."""

    def create_draft(state: ApprovalState) -> dict:
        step_print("📝", "DRAFT NODE", "Entering create_draft node...")
        print(f"   Request: \"{state['request']}\"")
        print("   Calling LLM to generate draft...")

        response = llm.invoke(f"Create a professional response for: {state['request']}")

        print(f"   Draft generated ({len(response.content.split())} words)")
        print(f"   Preview: {response.content[:100]}...")
        return {"draft": response.content}

    def wait_for_approval(state: ApprovalState) -> dict:
        step_print("👁️", "APPROVAL NODE", "Entering wait_for_approval node...")
        print(f"   Approved: {state['approved']}")
        print(
            f"   Feedback: '{state['feedback']}'"
            if state["feedback"]
            else "   Feedback: (none yet)"
        )
        # This node is where we'll interrupt
        return state

    def finalize(state: ApprovalState) -> dict:
        step_print("📦", "FINALIZE NODE", "Entering finalize node...")
        print(f"   Approved: {state['approved']}")

        if state["approved"]:
            print("   Action: Using draft as-is (human approved)")
            return {"final": state["draft"]}
        else:
            print("   Action: Revising draft based on feedback...")
            print(f"   Feedback: \"{state['feedback']}\"")
            # Incorporate feedback
            response = llm.invoke(
                f"Revise this draft based on feedback:\n\n"
                f"Draft: {state['draft']}\n\n"
                f"Feedback: {state['feedback']}"
            )
            print(f"   Revised draft generated ({len(response.content.split())} words)")
            return {"final": response.content}

    graph = StateGraph(ApprovalState)

    graph.add_node("draft", create_draft)
    graph.add_node("approval", wait_for_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "approval")
    graph.add_edge("approval", "finalize")
    graph.add_edge("finalize", END)

    # interrupt_before REQUIRES a checkpointer: pausing means persisting the state and
    # returning from invoke(), so there must be somewhere to store the frozen snapshot.
    # The process is not blocked — invoke() returns normally with the partial state.
    memory = MemorySaver()
    app = graph.compile(
        checkpointer=memory, interrupt_before=["approval"]  # Pause before this node
    )

    print("\n" + "=" * 55)
    print("  HUMAN-IN-THE-LOOP: APPROVAL WORKFLOW")
    print("=" * 55)

    print("\n   Graph: START -> [draft] -> ⏸️ -> [approval] -> [finalize] -> END")
    print("   Interrupt set BEFORE: 'approval' node")

    # Configuration for this thread
    config = {"configurable": {"thread_id": "demo-1"}}

    # ─── PHASE 1: Run until interrupt ───
    phase_banner(1, "RUN UNTIL INTERRUPT")
    print("   Calling app.invoke() with initial state...")
    print("   The graph will run until it hits the interrupt point.\n")

    result = app.invoke(
        {
            "request": "Write a thank-you email for a job interview",
            "draft": "",
            "approved": False,
            "feedback": "",
            "final": "",
        },
        config,
    )

    step_print("⏸️", "PAUSED", "Graph execution interrupted!")
    print(f"   Draft is ready: {result['draft'][:150]}...")
    print(f"   Final is empty: '{result['final']}'")
    print("\n   The graph is now FROZEN. Waiting for human input.")
    print("   In a real app, your frontend would show the draft here.")

    # ─── PHASE 2: Inspect paused state ───
    phase_banner(2, "INSPECT PAUSED STATE")

    current_state = app.get_state(config)
    print("   app.get_state(config) tells us:")
    print(f"   Next node(s): {current_state.next}")
    print(f"   State keys: {list(current_state.values.keys())}")
    print(f"   Draft filled: {'Yes' if current_state.values['draft'] else 'No'}")
    print(f"   Approved: {current_state.values['approved']}")
    print(f"   Final filled: {'Yes' if current_state.values['final'] else 'No'}")

    # ─── PHASE 3: Human provides feedback and resume ───
    phase_banner(3, "HUMAN INJECTS FEEDBACK + RESUME")

    feedback_text = (
        "Make it more concise and add specific mention of the company culture"
    )
    print("   Human decision: REJECT (request changes)")
    print(f'   Human feedback: "{feedback_text}"')
    print("\n   Calling app.update_state() to inject human input...")

    # update_state writes a partial dict into the paused checkpoint (applying the same
    # reducers a node would) and creates a new checkpoint — this is the supported way
    # to inject human decisions between steps.
    app.update_state(
        config, {"approved": False, "feedback": feedback_text}  # Request changes
    )

    print("   State updated. approved=False, feedback set.")
    print("\n   Calling app.invoke(None, config) to RESUME...")
    print("   (None means 'no new input, just continue from checkpoint')\n")

    # invoke(None, config) resumes from the checkpoint. Passing a real input dict here
    # would instead start a new step with that input merged in — not what you want.
    final_result = app.invoke(None, config)

    # ─── RESULT ───
    step_print("✅", "WORKFLOW COMPLETE", "")
    print(f"   Final result ({len(final_result['final'].split())} words):")
    print(f"   {final_result['final'][:200]}...")
    print("\n   Graph path taken:")
    print(
        "   START -> [draft] -> ⏸️ PAUSE -> human feedback -> [approval] -> [finalize] -> END"
    )


# ════════════════════════════════════════════════════════
# DEMO 2: Iterative Review (Human-in-the-Loop + Cycles)
# ════════════════════════════════════════════════════════


class ReviewState(TypedDict):
    document: str
    review_comments: list[str]
    revision_count: int
    status: str


def demo_iterative_review():
    """Multiple rounds of human review."""

    def submit_for_review(state: ReviewState) -> dict:
        step_print("📋", "SUBMIT NODE", f"Round {state['revision_count'] + 1}")
        print(f"   Status incoming: '{state['status']}'")
        print("   Setting status to 'pending_review'")
        print(f"   Document preview: {state['document'][:100]}...")
        return {"status": "pending_review"}

    def apply_feedback(state: ReviewState) -> dict:
        step_print(
            "🔧", "APPLY FEEDBACK NODE", f"Revision #{state['revision_count'] + 1}"
        )

        if not state["review_comments"]:
            print("   No comments to apply. Passing through.")
            return state

        feedback = state["review_comments"][-1]
        print(f'   Feedback to apply: "{feedback}"')
        print(f"   Current document: {state['document'][:80]}...")
        print("   Calling LLM to revise...")

        response = llm.invoke(
            f"Revise this document based on feedback:\n\n"
            f"Document: {state['document']}\n\n"
            f"Feedback: {feedback}"
        )

        print(f"   Revised document ({len(response.content.split())} words)")
        print(f"   Preview: {response.content[:100]}...")

        return {
            "document": response.content,
            "revision_count": state["revision_count"] + 1,
            "status": "revised",
        }

    # Routes purely on a status field that only a HUMAN ever sets to "approved" (via
    # update_state), so the revise loop can never terminate itself — the exit condition
    # is external input, not model judgement.
    def route_after_review(state: ReviewState) -> Literal["apply", "done"]:
        step_print("🔀", "ROUTER", f"Checking status: '{state['status']}'")
        if state["status"] == "approved":
            print("   Decision: APPROVED -> routing to 'done' node")
            return "done"
        print("   Decision: NOT APPROVED -> routing to 'apply' node")
        return "apply"

    def finalize(state: ReviewState) -> dict:
        step_print("🏁", "DONE NODE", "Finalizing document")
        print(f"   Total revisions: {state['revision_count']}")
        print(f"   Final document: {state['document'][:100]}...")
        return {"status": "finalized"}

    graph = StateGraph(ReviewState)

    graph.add_node("submit", submit_for_review)
    graph.add_node("apply", apply_feedback)
    graph.add_node("done", finalize)

    graph.add_edge(START, "submit")
    graph.add_conditional_edges(
        "submit", route_after_review, {"apply": "apply", "done": "done"}
    )
    graph.add_edge("apply", "submit")  # Loop for more reviews
    graph.add_edge("done", END)

    memory = MemorySaver()
    # Interrupting before a node that sits INSIDE the cycle means the pause fires on
    # every iteration — one human review per revision round.
    app = graph.compile(checkpointer=memory, interrupt_before=["submit"])

    print("\n" + "=" * 55)
    print("  HUMAN-IN-THE-LOOP: ITERATIVE REVIEW WORKFLOW")
    print("=" * 55)

    print("\n   Graph: START -> ⏸️ -> [submit] -> [ROUTER]")
    print("                                       ├── approved -> [done] -> END")
    print(
        "                                       └── else -> [apply] -> ⏸️ [submit] (LOOP)"
    )
    print("   Interrupt set BEFORE: 'submit' node (fires EVERY loop)")

    config = {"configurable": {"thread_id": "review-1"}}

    # ─── ROUND 0: Initial submission ───
    phase_banner(0, "INITIAL SUBMISSION")
    print("   Sending initial document into the graph...")
    print('   Document: "AI is technology that helps computers think."\n')

    result = app.invoke(
        {
            "document": "AI is technology that helps computers think.",
            "review_comments": [],
            "revision_count": 0,
            "status": "",
        },
        config,
    )

    step_print("⏸️", "PAUSED", "Graph hit interrupt_before='submit'")
    print(f"   Document ready for review: \"{result['document']}\"")
    print(f"   Revisions so far: {result['revision_count']}")

    current_state = app.get_state(config)
    print(f"   Next node: {current_state.next}")
    print("\n   Waiting for human reviewer...")

    # ─── ROUND 1: Reviewer wants changes ───
    phase_banner(1, "REVIEWER REQUESTS CHANGES")

    feedback_1 = "Add more technical depth and examples"
    print(f'   Reviewer says: "{feedback_1}"')
    print("   Reviewer sets status: 'needs_revision'")
    print("\n   Calling app.update_state() to inject review...")

    # review_comments has no reducer, so each update REPLACES the list rather than
    # appending — apply_feedback's [-1] therefore always sees the newest comment.
    app.update_state(
        config, {"review_comments": [feedback_1], "status": "needs_revision"}
    )

    print("   State updated. Calling app.invoke(None) to resume...\n")

    result = app.invoke(None, config)

    step_print("⏸️", "PAUSED AGAIN", "Graph looped back to 'submit' and paused")
    print(f"   Revised document: {result['document'][:150]}...")
    print(f"   Revisions so far: {result['revision_count']}")

    current_state = app.get_state(config)
    print(f"   Next node: {current_state.next}")
    print("\n   Waiting for human reviewer again...")

    # ─── ROUND 2: Reviewer wants more changes ───
    phase_banner(2, "REVIEWER REQUESTS MORE CHANGES")

    feedback_2 = "Good improvement! Now add a concrete example of neural networks"
    print(f'   Reviewer says: "{feedback_2}"')
    print("   Reviewer sets status: 'needs_revision'")

    app.update_state(
        config, {"review_comments": [feedback_2], "status": "needs_revision"}
    )

    print("   Resuming graph...\n")

    result = app.invoke(None, config)

    step_print("⏸️", "PAUSED AGAIN", "Graph looped back to 'submit' and paused")
    print(f"   Revised document: {result['document'][:150]}...")
    print(f"   Revisions so far: {result['revision_count']}")

    # ─── ROUND 3: Reviewer approves ───
    phase_banner(3, "REVIEWER APPROVES")

    print('   Reviewer says: "Looks great!"')
    print("   Reviewer sets status: 'approved'")

    app.update_state(config, {"status": "approved"})

    print("   Resuming graph for final time...\n")

    final = app.invoke(None, config)

    # ─── FINAL SUMMARY ───
    step_print("✅", "WORKFLOW COMPLETE", "")
    print(f"   Final status: {final['status']}")
    print(f"   Total revisions: {final['revision_count']}")
    print(f"   Final document: {final['document'][:200]}...")
    print("\n   Full timeline:")
    print("   Round 0: START -> ⏸️ (human reviews initial doc)")
    print("   Round 1: resume -> [submit] -> [apply] -> ⏸️ (human reviews revision 1)")
    print("   Round 2: resume -> [submit] -> [apply] -> ⏸️ (human reviews revision 2)")
    print("   Round 3: resume -> [submit] -> [done] -> END (human approved!)")


if __name__ == "__main__":
    print("=" * 55)
    print("  Demo 1: Interrupt for Approval")
    print("=" * 55)
    demo_interrupt_for_approval()

    print("\n" + "=" * 55)
    print("  Demo 2: Iterative Review")
    print("=" * 55)
    demo_iterative_review()
