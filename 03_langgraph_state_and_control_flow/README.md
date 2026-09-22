# 02 — LangGraph Control Flow

> **Numeric prefixes = suggested reading order.** Mostly increasing difficulty, but two are genuine dependencies: `04_cycles_loops.py` builds on the conditional-edge routing from `03`, and `06_human_in_loop.py` requires the checkpointer from `05_checkpointing.py` (interrupts don't work without one).

LangGraph fundamentals: modelling an agent as a `StateGraph` of nodes and edges over a typed state object, then adding branching, loops, persistence, human approval, and failure handling.

The conceptual overview also explains how to choose between lightweight `TypedDict` graph state and runtime-validated Pydantic `BaseModel` state, including the recommended production API-boundary pattern: [TypedDict vs. Pydantic BaseModel](../docs/langgraph-overview.md#choosing-a-state-schema-typeddict-vs-pydantic-basemodel).

## `01_langgraph_core.py`

**What it does** — the primitives:
- `StateGraph(TypedDict)` + `add_node` / `add_edge(START, ...)` / `.compile()`.
- State reducers via `Annotated[list[str], operator.add]` and `Annotated[int, operator.add]`, so parallel/sequential writes merge instead of overwrite.
- The canonical `Annotated[list[BaseMessage], add_messages]` chat state.
- A linear 3-node analyze→enhance→finalize pipeline.
- Visualization with `app.get_graph().draw_mermaid()` / `.draw_mermaid_png()`.

**What it's for**
- Knowing when to use a graph over a plain chain — multi-step workflows where each step reads and updates shared state.
- Keeping the topology inspectable.

## `02_first_graph.py`

**What it does** — a minimal two-node conversation graph:
- `analyze_sentiment` classifies the last message.
- `generate_response` picks a different system prompt per sentiment bucket.
- State is a `ConversationState` TypedDict with an `operator.add` message list plus scalar `sentiment`/`response_count`.

**What it's for**
- The "hello world" of stateful agents — showing that one node's output steers the next node's behaviour through state.

## `03_conditional_edges.py`

**What it does** — routing via `add_conditional_edges(source, router_fn, {label: node})`, where the router returns a `Literal`. Three demos:
- Intent classification into question/command/statement handlers.
- A quality loop routing `evaluate → improve → evaluate` until score ≥ 7 or 3 iterations.
- 2×2 multi-path routing on urgency × complexity.

**What it's for**
- Triage and dispatch — support ticket routing, intent classification.
- Any "pick the right handler" step where an `if` needs to be part of the graph, not buried in a node.

## `04_cycles_loops.py`

**What it does** — self-correcting cycles:
- A code generator that loops `generate → validate → generate`, where `validate` actually `compile()`s and `exec()`s the code against test cases and pushes failures into an `Annotated[list[str], operator.add]` errors field.
- An iterative research graph that alternates research/question-generation until `max_depth`, then synthesizes.

**What it's for**
- Reflection/refinement agents — anything where the model's first output must be checked by real feedback (compiler, tests, evaluator) and retried.
- Enforcing an explicit iteration cap to prevent runaway loops.

## `05_checkpointing.py`

**What it does** — persistence:
- `MemorySaver` for dev and `SqliteSaver.from_conn_string(path)` for durable storage, both passed as `graph.compile(checkpointer=...)` and addressed by `config={"configurable": {"thread_id": ...}}`.
- State inspection with `app.get_state()` (`.values`, `.next`, `.config`, `.parent_config`, `.metadata`, `.created_at`).
- `app.get_state_history()` for time travel.
- `app.update_state()` to fork a thread, and resuming from a specific `checkpoint_id`.

**What it's for**
- Multi-turn memory that survives restarts.
- Per-user conversation isolation via `thread_id`.
- Debugging/rewind — replaying a run from the checkpoint before a bad step.

## `06_human_in_loop.py`

**What it does** — interrupts:
- `graph.compile(checkpointer=memory, interrupt_before=["approval"])` freezes execution.
- The human inspects via `get_state`, injects a decision with `app.update_state(config, {...})`, and resumes with `app.invoke(None, config)`.
- A second demo puts the interrupt inside a cycle, so review fires every revision round until status is `approved`.

**What it's for**
- Approval gates and review workflows — draft-then-approve email/content flows.
- Any action too consequential to run unattended.

## `07_error_handling.py`

**What it does** — reliability patterns, mostly framework-independent:
- A `with_retry` decorator (exponential backoff + jitter).
- A `CircuitBreaker` class (closed/open/half-open with recovery timeout).
- A `FallbackChain` trying gpt-4o-mini → gpt-4o → Claude, with an in-process response cache and `@traceable` for LangSmith.
- A LangGraph agent whose conditional edge routes `retry / error / success` off a `retry_count` in state.

**What it's for**
- Surviving flaky providers in production — rate limits, timeouts, and outages — without cascading failures or losing the user's request.
