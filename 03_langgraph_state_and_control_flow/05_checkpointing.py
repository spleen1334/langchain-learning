"""
Checkpointing and Persistence in LangGraph
Save and resume agent state
"""

import operator
import tempfile
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def demo_memory_saver():
    """In-memory checkpointing for development."""

    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)

    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # A checkpointer makes the graph stateful ACROSS invoke() calls. MemorySaver keeps
    # everything in a process-local dict — fine for dev, lost on restart.
    saver = MemorySaver()
    app = graph.compile(checkpointer=saver)

    # thread_id is the conversation key: the same id resumes the same state, a different
    # id starts a clean one. Passing no config at all with a checkpointer is an error.
    config: RunnableConfig = {"configurable": {"thread_id": "user-123"}}

    print("Memory Saver Demo (Multi-turn conversation):\n")

    # Turn 1
    result = app.invoke(
        {"messages": [HumanMessage(content="My name is Paulo")]}, config
    )
    print(f"Turn 1 - AI: {result['messages'][-1].content}")

    # Only the NEW message is passed in — the checkpointer reloads the prior messages
    # and the operator.add reducer appends this one, so the LLM sees the full history.
    result = app.invoke({"messages": [HumanMessage(content="What's my name?")]}, config)
    print(f"Turn 2 - AI: {result['messages'][-1].content}")

    # Check full history
    state = app.get_state(config)
    print(f"\nTotal messages in state: {len(state.values['messages'])}")


def demo_sqlite_persistence():
    """SQLite persistence for durable storage."""

    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print("\nSQLite Persistence Demo:")
    print(f"Database: {db_path}\n")

    # from_conn_string is a context manager because it owns the DB connection; the graph
    # must be compiled INSIDE the `with` or the checkpointer's connection is already closed.
    # First session
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = graph.compile(checkpointer=saver)
        config: RunnableConfig = {"configurable": {"thread_id": "persistent-user"}}

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content="Remember: The secret code is ALPHA-7")
                ]
            },
            config,
        )
        print("Session 1 - Stored secret code")

        # PostgresSaver with a real database!
        # Simulate app restart - new session
    # Fresh saver, fresh compile, SAME db file + SAME thread_id => the conversation is
    # recovered from disk. This is the whole point of SqliteSaver over MemorySaver.
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = graph.compile(checkpointer=saver)
        config: RunnableConfig = {"configurable": {"thread_id": "persistent-user"}}

        result = app.invoke(
            {"messages": [HumanMessage(content="What was the secret code?")]}, config
        )
        print(f"Session 2 - AI: {result['messages'][-1].content}")


def demo_state_inspection():
    """Inspect and manipulate checkpoint state."""

    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config: RunnableConfig = {"configurable": {"thread_id": "inspect-demo"}}

    print("\nState Inspection Demo:\n")

    # Build up some state
    app.invoke({"messages": [HumanMessage(content="Hello!")]}, config)
    app.invoke({"messages": [HumanMessage(content="How are you?")]}, config)

    # Get current state
    state = app.get_state(config)

    print("Current state:")
    print(f"  Next node: {state.next}")
    print(f"  Message count: {len(state.values['messages'])}")

    # get_state_history yields snapshots NEWEST-FIRST, so checkpoint 0 here is the most
    # recent one; the loop breaks early because a long thread has many checkpoints.
    print("\nState history:")
    for i, snapshot in enumerate(app.get_state_history(config)):
        print(f"  Checkpoint {i}: {len(snapshot.values['messages'])} messages")
        if i >= 3:
            print("  ...")
            break


def demo_branching_conversations():
    """Branch conversations from checkpoints."""

    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    print("\nBranching Conversations Demo:\n")

    # Main conversation
    main_config: RunnableConfig = {"configurable": {"thread_id": "main"}}
    app.invoke(
        {"messages": [HumanMessage(content="What's the weather like?")]}, main_config
    )

    # Get checkpoint to branch from
    main_state = app.get_state(main_config)

    # Branch A - Beach vacation
    branch_a_config: RunnableConfig = {"configurable": {"thread_id": "branch-beach"}}
    # Forking = writing the parent thread's state into a NEW thread_id. Both branches
    # then evolve independently and neither can affect "main".
    # Copy state to new thread
    app.update_state(branch_a_config, main_state.values)

    result_a = app.invoke(
        {"messages": [HumanMessage(content="What about a beach vacation?")]},
        branch_a_config,
    )
    print(f"Branch A (Beach): {result_a['messages'][-1].content[:100]}...")

    # Branch B - Mountain adventure
    branch_b_config: RunnableConfig = {"configurable": {"thread_id": "branch-mountain"}}
    app.update_state(branch_b_config, main_state.values)

    result_b = app.invoke(
        {"messages": [HumanMessage(content="What about mountain hiking?")]},
        branch_b_config,
    )
    print(f"Branch B (Mountain): {result_b['messages'][-1].content[:100]}...")


def demo_checkpoint_internals():
    """
    Peek inside a checkpoint — see exactly what LangGraph saves.

    Uses a 2-node graph so we generate multiple checkpoints,
    then walks through every field in the checkpoint object.
    """

    # ── Build a 2-node graph so we get several checkpoints ──

    class TaskState(TypedDict):
        messages: Annotated[list[BaseMessage], operator.add]
        step: str

    def analyze(state: TaskState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response], "step": "analyzed"}

    def summarize(state: TaskState) -> dict:
        summary_prompt = [
            HumanMessage(
                content=f"Summarize this in one sentence: {state['messages'][-1].content}"
            )
        ]
        response = llm.invoke(summary_prompt)
        return {"messages": [response], "step": "summarized"}

    graph = StateGraph(TaskState)
    graph.add_node("analyze", analyze)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "summarize")
    graph.add_edge("summarize", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config: RunnableConfig = {"configurable": {"thread_id": "internals-demo"}}

    print("\nCheckpoint Internals Demo")
    print("=" * 55)
    print("Graph: START -> [analyze] -> [summarize] -> END")
    print("=" * 55)

    # ── Run the graph ──

    app.invoke(
        {"messages": [HumanMessage(content="Explain why the sky is blue")], "step": ""},
        config,
    )

    # ════════════════════════════════════════════════════════
    # PART 1: What's in the CURRENT state snapshot?
    # ════════════════════════════════════════════════════════

    print("\n--- PART 1: Current State Snapshot (app.get_state) ---\n")

    state = app.get_state(config)

    # state.values — your actual TypedDict data
    print("1) state.values (your state data):")
    print(f"   step: '{state.values['step']}'")
    print(f"   messages: {len(state.values['messages'])} total")
    for i, msg in enumerate(state.values["messages"]):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"     [{i}] {role}: {msg.content[:80]}...")

    # state.next — which node runs next (empty = graph finished)
    print("\n2) state.next (pending node):")
    print(f"   {state.next if state.next else '() — graph finished, no pending nodes'}")

    # state.config — the config that produced this snapshot
    # "configurable" is a NotRequired key on RunnableConfig, so .get(..., {}) rather
    # than [...] — it's always populated here, but the type checker can't know that.
    state_configurable = state.config.get("configurable", {})
    print("\n3) state.config (thread + checkpoint IDs):")
    print(f"   thread_id:     {state_configurable.get('thread_id')}")
    print(f"   checkpoint_id: {state_configurable.get('checkpoint_id')}")

    # state.metadata — who created this checkpoint. It's Optional, so a run with no
    # metadata (shouldn't happen once a checkpointer is attached) falls back to {}.
    metadata = state.metadata or {}
    print("\n4) state.metadata (provenance info):")
    print(f"   source:  {metadata.get('source', 'N/A')}")
    print(f"   step:    {metadata.get('step', 'N/A')}")
    print(f"   writes:  {metadata.get('writes', 'N/A')}")

    # state.parent_config — pointer to the PREVIOUS checkpoint
    print("\n5) state.parent_config (previous checkpoint):")
    if state.parent_config:
        parent_configurable = state.parent_config.get("configurable", {})
        print(f"   parent checkpoint_id: {parent_configurable.get('checkpoint_id')}")
    else:
        print("   None — this is the very first checkpoint")

    # state.created_at — timestamp
    print("\n6) state.created_at (when saved):")
    print(f"   {state.created_at}")

    # ════════════════════════════════════════════════════════
    # PART 2: Walk through ALL checkpoints (time travel)
    # ════════════════════════════════════════════════════════

    print("\n--- PART 2: Full Checkpoint History (app.get_state_history) ---\n")
    print("LangGraph saves a checkpoint at EACH step. Let's see them all:\n")

    for i, snapshot in enumerate(app.get_state_history(config)):
        snapshot_metadata = snapshot.metadata or {}
        step_num = snapshot_metadata.get("step", "?")
        source = snapshot_metadata.get("source", "?")
        writes = snapshot_metadata.get("writes", {})
        msg_count = len(snapshot.values.get("messages", []))
        checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
        current_step = snapshot.values.get("step", "")

        # metadata["writes"] is keyed by node name, so its first key identifies the
        # node that produced this checkpoint (empty for the initial input checkpoint).
        node_name = next(iter(writes.keys())) if writes else "—"

        print(f"  Checkpoint {i}:")
        print(f"    id:         {str(checkpoint_id)[:30]}...")
        print(f"    source:     {source}")
        print(f"    step:       {step_num}")
        print(f"    written by: {node_name}")
        print(f"    state.step: '{current_step}'")
        print(f"    messages:   {msg_count}")
        print(f"    next:       {snapshot.next if snapshot.next else '() — finished'}")
        print(f"    created_at: {snapshot.created_at}")
        print()

    # ════════════════════════════════════════════════════════
    # PART 3: Jump to a specific checkpoint (rewind)
    # ════════════════════════════════════════════════════════

    print("--- PART 3: Rewind — Jump to a Previous Checkpoint ---\n")

    # Find the checkpoint right after the "analyze" node ran
    target_snapshot = None
    for snapshot in app.get_state_history(config):
        writes = (snapshot.metadata or {}).get("writes", {})
        if "analyze" in writes:
            target_snapshot = snapshot
            break

    if target_snapshot:
        target_id = target_snapshot.config.get("configurable", {}).get("checkpoint_id")
        print(f"  Found checkpoint after 'analyze' node: {str(target_id)[:30]}...")
        print(f"  Messages at that point: {len(target_snapshot.values['messages'])}")
        print(f"  state.step at that point: '{target_snapshot.values.get('step', '')}'")

        # You can resume from this exact checkpoint
        # Adding checkpoint_id to the config addresses ONE specific snapshot instead of
        # the thread's latest — this is how time travel / replay works.
        rewind_config: RunnableConfig = {
            "configurable": {"thread_id": "internals-demo", "checkpoint_id": target_id}
        }

        rewound_state = app.get_state(rewind_config)
        print(f"\n  Loaded checkpoint — next node would be: {rewound_state.next}")
        print("  We're back to BEFORE 'summarize' ran!")
        print(
            "  Calling invoke(None) from here would re-run 'summarize' with fresh output."
        )
    else:
        print("  Could not find target checkpoint.")

    # ════════════════════════════════════════════════════════
    # SUMMARY: Anatomy of a checkpoint
    # ════════════════════════════════════════════════════════

    print("\n" + "=" * 55)
    print("  CHECKPOINT ANATOMY — What Gets Saved")
    print("=" * 55)
    print(
        """
    state.values        → Your TypedDict data (messages, step, etc.)
    state.next          → Tuple of nodes that run next (() if done)
    state.config        → thread_id + checkpoint_id (unique address)
    state.parent_config → Previous checkpoint's address (linked list)
    state.metadata      → source, step number, which node wrote
    state.created_at    → Timestamp of when this checkpoint was saved

    Checkpoints are saved:
      1. BEFORE the first node runs (initial input state)
      2. AFTER each node completes (with updated state)
      3. At interrupt points (frozen state for human-in-the-loop)

    Think of it as a linked list of snapshots:
      [initial] --> [after analyze] --> [after summarize]
         ^               ^                    ^
       parent          parent              current (latest)
    """
    )


if __name__ == "__main__":
    # demo_memory_saver()
    # demo_sqlite_persistence()
    # demo_state_inspection()
    # demo_branching_conversations()
    demo_checkpoint_internals()
