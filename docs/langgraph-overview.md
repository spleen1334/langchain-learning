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

## LangChain vs. LangGraph

They're layered, not competing — LangGraph is orchestration built *on top of* LangChain, not a replacement for it. A `StateGraph` node still calls `init_chat_model`, still passes around `HumanMessage`/`AIMessage`, still binds `@tool`-decorated functions; LangGraph adds the state machine around those LangChain primitives, it doesn't reimplement them.

Why not just use LangGraph for everything, then:
- **A single call or a straight pipeline doesn't need a state machine.** `prompt | model | parser` is one line; the equivalent single-node `StateGraph` is boilerplate — a state schema, a node function, two edges, a compile step — for the same result.
- **LangGraph has no integrations of its own.** Document loaders, embeddings, vector stores, retrievers — all LangChain (or its provider packages). You reach into LangChain for these regardless of which orchestration layer wraps them.
- **LCEL is the easier read when the flow really is linear.** Save the graph's nodes/edges/reducers for when the control flow is not a straight line — branching, loops, pausing, or persisted state across turns are the actual trigger, per the list above.

Rule of thumb: default to a LangChain chain; reach for `StateGraph` the moment the flow needs to branch, loop, persist, or pause. See [langchain-overview.md § When LangChain stops being enough](langchain-overview.md#when-langchain-stops-being-enough).

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

#### Choosing a state schema: `TypedDict` vs. Pydantic `BaseModel`

LangGraph accepts `TypedDict`, a dataclass, or a Pydantic `BaseModel` as its state schema. Use `TypedDict` as the default for internal graph state; use `BaseModel` when runtime validation of the graph's initial input is worth the extra cost.

| | `TypedDict` | Pydantic `BaseModel` |
|---|---|---|
| Type checking | Static only; no runtime enforcement | Runtime parsing and validation |
| State access in a node | `state["query"]` | `state.query` |
| Constraints and custom validation | Not built in | `Field(...)`, nested models, and validators |
| Type coercion | None | May coerce compatible input values unless configured strictly |
| Runtime cost | Very low; state remains a plain dictionary | Higher; Pydantic constructs models and recursively validates fields |
| Best fit | Trusted internal workflow state | Complex or untrusted input at a validation boundary |

The performance difference comes from work done at runtime. A `TypedDict` disappears at runtime: it describes a dictionary to the type checker but does not inspect or transform values. Pydantic must construct a model, traverse fields and nested models, parse or coerce values, run constraints and validators, and build detailed errors. That cost is usually negligible beside an LLM or network call, but it can become noticeable for large or deeply nested state, high-throughput graphs, or nodes that execute repeatedly in loops.

Pydantic state also has important LangGraph limitations:

- validation occurs on input to the first node, not on every subsequent node update or graph output;
- the normal graph result is still a dictionary rather than a guaranteed `BaseModel` instance;
- a validation traceback does not necessarily identify the graph node responsible for the bad value;
- the higher-level LangChain `create_agent` API does not support Pydantic state schemas.

Therefore, using `BaseModel` as graph state does **not** make every state transition runtime-safe. Nodes should still return correct partial updates, and important invariants may need explicit validation where they are produced.

For a production API, keep external contracts and internal workflow state separate:

```text
API request BaseModel
        ↓ validate untrusted input once
LangGraph TypedDict state
        ↓ lightweight internal state updates
API response BaseModel
        ↓ validate the public response contract
HTTP response
```

```python
class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    max_results: int = Field(default=5, ge=1, le=20)


class ResearchState(TypedDict):
    query: str
    max_results: int
    findings: Annotated[list[str], operator.add]
    answer: str


class ResearchResponse(BaseModel):
    answer: str
    source_count: int = Field(ge=0)


request = ResearchRequest.model_validate(request_body)
result = graph.invoke(request.model_dump())
response = ResearchResponse(
    answer=result["answer"],
    source_count=len(result["findings"]),
)
```

This pattern gives the API strong validation where data crosses a trust boundary while keeping frequently updated graph state simple and fast. Reach for Pydantic graph state instead when nodes genuinely benefit from validated nested objects or attribute access and the graph's workload is not performance-sensitive.

Reference: [LangGraph Graph API — state schemas and Pydantic limitations](https://docs.langchain.com/oss/python/langgraph/use-graph-api#use-pydantic-models-for-graph-state).

→ [`03_langgraph_state_and_control_flow/01_langgraph_core.py`](../03_langgraph_state_and_control_flow/01_langgraph_core.py)

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

→ [`03_langgraph_state_and_control_flow/02_first_graph.py`](../03_langgraph_state_and_control_flow/02_first_graph.py), [`04_multi_agent_systems/04_parallel_agents.py`](../04_multi_agent_systems/04_parallel_agents.py)

### Conditional edges
```python
def router(state) -> Literal["a", "b"]: ...
graph.add_conditional_edges("source", router, {"a": "node_a", "b": "node_b"})
```
- **The router is plain Python** — it can read state set by an LLM classifier node, or call an LLM itself.
- **Pair it with `with_structured_output(SomeLiteralSchema)`** — the reliable way to get an LLM to choose a branch.

→ [`03_langgraph_state_and_control_flow/03_conditional_edges.py`](../03_langgraph_state_and_control_flow/03_conditional_edges.py)

### Cycles
- **An edge pointing backwards makes a loop** — `generate → validate → (conditional) → generate`.
- **Always carry an `iteration` counter in state** and terminate on `iteration >= max` in the router — nothing else stops an infinite loop.
- **Backstop** — LangGraph also enforces a global recursion limit.

→ [`03_langgraph_state_and_control_flow/04_cycles_loops.py`](../03_langgraph_state_and_control_flow/04_cycles_loops.py)

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

Unlike `RunnableWithMessageHistory`, which stores and injects a chat transcript, a checkpointer saves every graph-state field plus execution position and checkpoint metadata. See [Conversation History vs. LangGraph Checkpointing](conversation-history-vs-checkpointing.md) for the focused comparison.

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

→ [`03_langgraph_state_and_control_flow/05_checkpointing.py`](../03_langgraph_state_and_control_flow/05_checkpointing.py)

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

→ [`03_langgraph_state_and_control_flow/06_human_in_loop.py`](../03_langgraph_state_and_control_flow/06_human_in_loop.py)

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

→ [`03_langgraph_state_and_control_flow/07_error_handling.py`](../03_langgraph_state_and_control_flow/07_error_handling.py)

## Where each concept lives

| Concept | File |
|---|---|
| StateGraph, nodes, edges, reducers, mermaid export | `03_langgraph_state_and_control_flow/01_langgraph_core.py` |
| Minimal two-node stateful graph | `03_langgraph_state_and_control_flow/02_first_graph.py` |
| Conditional edges / routing | `03_langgraph_state_and_control_flow/03_conditional_edges.py` |
| Cycles, self-correction, iteration caps | `03_langgraph_state_and_control_flow/04_cycles_loops.py` |
| Checkpointers, state inspection, time travel, branching threads | `03_langgraph_state_and_control_flow/05_checkpointing.py` |
| `interrupt_before`, `update_state`, `invoke(None)` | `03_langgraph_state_and_control_flow/06_human_in_loop.py` |
| Retry / circuit breaker / fallback / in-graph error routing | `03_langgraph_state_and_control_flow/07_error_handling.py` |
| `ToolNode`, agent loop | `04_multi_agent_systems/01_tool_calling_agent.py` |
| Subgraphs, parallel fan-out/fan-in | `04_multi_agent_systems/07_hierarchical_agents.py`, `04_parallel_agents.py` |
| Send API, streaming, quality-gate loop | `projects/03_multi_agent_research_system.py` |
