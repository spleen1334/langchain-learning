# LangGraph Overview

## What it is

LangGraph models an application as a **state machine**. Where an LCEL chain is a one-shot DAG, a LangGraph graph can branch, loop, pause, persist, and resume.

The three pieces:
- **State** — a typed state object.
- **Nodes** — functions that read state and return partial updates.
- **Edges** — decide what runs next.

Use it when you need any of:
- Cycles (retry/refine)
- Conditional routing
- Durable multi-turn state
- Human approval mid-run
- Multiple agents coordinating

## Core concepts

### State
A `TypedDict` schema. Each node returns a *partial* dict; LangGraph merges it. By default a key is overwritten — to accumulate, annotate it with a **reducer**:

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]   # smart message merge
    findings: Annotated[list[dict], operator.add]          # list concatenation
    count:    Annotated[int, operator.add]                 # summation
    status:   str                                          # last write wins
```

- **`add_messages`** (from `langgraph.graph`) is the one to reach for with chat state: it appends, and handles message IDs/updates rather than blindly concatenating.
- **Reducers make parallel branches safe** — two nodes writing the same key concurrently would otherwise conflict.

→ [`02_langgraph_control_flow/01_langgraph_core.py`](../02_langgraph_control_flow/01_langgraph_core.py)

### Nodes and edges
```python
graph = StateGraph(State)
graph.add_node("analyze", analyze_fn)
graph.add_edge(START, "analyze")
graph.add_edge("analyze", END)
app = graph.compile()
```
- **A node** is any callable `state -> dict`.
- **A compiled graph is itself a Runnable** (`.invoke`, `.stream`, `.batch`) — which is why a compiled subgraph can be dropped in as a node of a parent graph.
- **Multiple edges out of one node** = parallel fan-out.
- **Multiple edges into one node** = fan-in; that node waits for all of them.

→ [`02_langgraph_control_flow/02_first_graph.py`](../02_langgraph_control_flow/02_first_graph.py), [`04_multi_agent_systems/04_parallel_agents.py`](../04_multi_agent_systems/04_parallel_agents.py)

### Conditional edges
```python
def router(state) -> Literal["a", "b"]: ...
graph.add_conditional_edges("source", router, {"a": "node_a", "b": "node_b"})
```
- **The router is plain Python** — it can read state set by an LLM classifier node, or call an LLM itself.
- **Pair it with `with_structured_output(SomeLiteralSchema)`** — the reliable way to get an LLM to choose a branch.

→ [`02_langgraph_control_flow/03_conditional_edges.py`](../02_langgraph_control_flow/03_conditional_edges.py)

### Cycles
- **An edge pointing backwards makes a loop** — `generate → validate → (conditional) → generate`.
- **Always carry an `iteration` counter in state** and terminate on `iteration >= max` in the router — nothing else stops an infinite loop.
- **Backstop** — LangGraph also enforces a global recursion limit.

→ [`02_langgraph_control_flow/04_cycles_loops.py`](../02_langgraph_control_flow/04_cycles_loops.py)

### The Send API (dynamic fan-out)
- **When to use** — the number of parallel branches is only known at runtime.
- **How** — return a list of `Send(node_name, sub_state)` from a conditional edge.
- **What happens** — each `Send` spawns an independent invocation of that node; results merge back through reducers.

```python
def dispatch(state) -> list[Send]:
    return [Send("search_agent", {"search_query": q, "findings": []})
            for q in state["search_queries"]]
```

→ [`projects/03_multi_agent_research_system.py`](../projects/03_multi_agent_research_system.py)

### Checkpointing and persistence
`graph.compile(checkpointer=...)` saves a snapshot before the first node and after every node. Threads are addressed by config:

```python
config = {"configurable": {"thread_id": "user-123"}}
app.invoke({...}, config)   # turn 1
app.invoke({...}, config)   # turn 2 — sees turn 1's state
```

Checkpointers:
- `MemorySaver` — dev
- `SqliteSaver.from_conn_string(path)` — local durable
- `PostgresSaver` — production

A checkpoint holds:
- `values` — your state
- `next` — pending nodes
- `config` — thread_id + checkpoint_id
- `parent_config` — previous checkpoint; it's a linked list
- `metadata` — source, step, which node wrote
- `created_at`

Useful operations:
- `app.get_state(config)` — current snapshot
- `app.get_state_history(config)` — every checkpoint, newest first (time travel)
- `app.update_state(config, {...})` — write into state from outside the graph
- passing a `checkpoint_id` in config — resume/rewind to an exact point

→ [`02_langgraph_control_flow/05_checkpointing.py`](../02_langgraph_control_flow/05_checkpointing.py)

### Human-in-the-loop
Interrupts are checkpointing plus a pause:

```python
app = graph.compile(checkpointer=memory, interrupt_before=["approval"])
app.invoke(initial, config)              # runs until frozen
app.get_state(config)                    # show the draft to the human
app.update_state(config, {"feedback": ...})  # inject the decision
app.invoke(None, config)                 # None = resume from checkpoint
```
- **`interrupt_after=[...]`** also exists.
- **Put the interrupt inside a cycle** and it fires on every iteration — that's a review loop.

→ [`02_langgraph_control_flow/06_human_in_loop.py`](../02_langgraph_control_flow/06_human_in_loop.py)

### Tool execution
- **`ToolNode(tools)`** from `langgraph.prebuilt` executes whatever the model requested in `AIMessage.tool_calls` and appends `ToolMessage`s.
- **The standard agent loop** is `agent → (tool_calls?) → tools → agent`.

→ [`04_multi_agent_systems/01_tool_calling_agent.py`](../04_multi_agent_systems/01_tool_calling_agent.py)

### Subgraphs
- **How** — a compiled graph added via `add_node("team", compiled_subgraph)`.
- **Shared state schema** means the subgraph reads and writes the parent's state directly.
- **Why** — this is how hierarchical multi-agent systems are built.

→ [`04_multi_agent_systems/07_hierarchical_agents.py`](../04_multi_agent_systems/07_hierarchical_agents.py)

### Streaming and visualization
- `app.stream(state, stream_mode="updates")` — per-node state deltas as they complete (also `"values"`, `"messages"`)
- `app.get_graph().draw_mermaid()` / `.draw_mermaid_png()` — topology diagram; exports live in [`assets/graphs/`](../assets/graphs/)

### Error handling
- **Nothing is built in** beyond retries at the node level.
- **The repo's approach** — catch inside the node, write an `error` / `retry_count` into state, and let a conditional edge choose retry vs fallback vs give-up.
- **Around external calls** — wrap them in retry decorators, circuit breakers, or model fallback chains.

→ [`02_langgraph_control_flow/07_error_handling.py`](../02_langgraph_control_flow/07_error_handling.py)

## Where each concept lives

| Concept | File |
|---|---|
| StateGraph, nodes, edges, reducers, mermaid export | `02_langgraph_control_flow/01_langgraph_core.py` |
| Minimal two-node stateful graph | `02_langgraph_control_flow/02_first_graph.py` |
| Conditional edges / routing | `02_langgraph_control_flow/03_conditional_edges.py` |
| Cycles, self-correction, iteration caps | `02_langgraph_control_flow/04_cycles_loops.py` |
| Checkpointers, state inspection, time travel, branching threads | `02_langgraph_control_flow/05_checkpointing.py` |
| `interrupt_before`, `update_state`, `invoke(None)` | `02_langgraph_control_flow/06_human_in_loop.py` |
| Retry / circuit breaker / fallback / in-graph error routing | `02_langgraph_control_flow/07_error_handling.py` |
| `ToolNode`, agent loop | `04_multi_agent_systems/01_tool_calling_agent.py` |
| Subgraphs, parallel fan-out/fan-in | `04_multi_agent_systems/07_hierarchical_agents.py`, `04_parallel_agents.py` |
| Send API, streaming, quality-gate loop | `projects/03_multi_agent_research_system.py` |
