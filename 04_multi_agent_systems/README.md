# 04 — Multi-Agent Systems

> **Numeric prefixes = suggested reading order** (increasing difficulty; each file runs standalone). `01_tool_calling_agent.py` is the real prerequisite — every later pattern assumes the agent/tool loop. `03_multi_agent.py` is a flatter rewrite of the `02` supervisor, and `07_hierarchical_agents.py` composes the supervisor pattern into subgraphs, so read `02` first.

Architectures for splitting work across multiple specialized LLM agents in LangGraph: tool use, supervisors, hierarchies, parallel fan-out, handoffs, and the communication mechanisms underneath all of them.

## `01_tool_calling_agent.py`

**What it does** — the ReAct loop as a graph:
- `@tool`-decorated Python functions (`calculate`, `get_weather`, `search_web`, `divide`).
- `llm.bind_tools(tools)` and the prebuilt `ToolNode(tools)`.
- A conditional edge on `last_message.tool_calls` routing `agent → tools → agent` until the model stops requesting tools.
- A full trace walkthrough of `AIMessage.tool_calls` / `ToolMessage`, plus tool-level error returns.

**What it's for**
- The foundation of every agent — letting the model act on the world (APIs, calculators, search) instead of just producing text.

## `02_supervisor_agent.py`

**What it does** — supervisor/router architecture:
- A supervisor node uses `with_structured_output(RouteDecision)` (`Literal["researcher","writer","critic","FINISH"]` + reasoning) to pick the next specialist.
- Each specialist edges back to the supervisor.
- `FINISH` flips `task_complete` and routes to `finalize`.
- State uses `add_messages` so every agent sees the full transcript.

**What it's for**
- Coordinating a small team where the orchestration order isn't known in advance and can revisit agents (writer → critic → writer).

## `03_multi_agent.py`

**What it does** — the same supervisor pattern written top-level:
- Module-scope nodes with an `operator.add` message reducer.
- A spelled-out workflow prompt: researcher → writer → critic → back to writer if changes requested → FINISH.
- `finalize` scans backwards for the last `[Writer]` message as the deliverable.

**What it's for**
- A more readable/hackable version of the supervisor system — the one to start from when building your own team.

## `04_parallel_agents.py`

**What it does** — fan-out/fan-in:
- Three edges from `START` to research/creative/technical nodes that each write a *different* state field.
- All three edge into a `synthesize` node (LangGraph runs the branch superstep concurrently and merges).
- Also a map-reduce summarizer node pair (`map` over documents → `reduce` into one overview).

**What it's for**
- Latency wins when subtasks are independent.
- Multi-perspective analysis where you want distinct viewpoints rather than one averaged answer.

## `05_agent_handoffs.py`

**What it does** — triage and transfer:
- A `HandoffDecision` Pydantic schema (`handoff_to` ∈ sales/support/billing/stay/end, `reason`, `context`) produced via `with_structured_output`.
- The triage node writes `current_agent` + `context_summary` into state.
- A conditional edge dispatches to the specialist, and each specialist reads the handoff context in its system prompt.

**What it's for**
- Customer-service style routing.
- Critically, passing *context* along with control so the specialist doesn't re-interrogate the user.

## `06_agent_communication.py`

**What it does** — three ways agents share information:
- **Message passing** — researcher → fact-checker → summarizer each append a labelled `AIMessage` to an `add_messages` list that the next agent reads.
- **Shared typed fields** — `SharedFieldsState` where each agent writes its own field (`raw_data`, `analysis`, `confidence_score`, `recommendations`).
- **Blackboard** — a shared workspace of `drafts` / `critiques` (both `operator.add`) with drafter/critic looping via a conditional edge until a structured `ApprovalDecision` approves or iteration ≥ 3.

**What it's for** — choosing your coordination substrate:
- Messages when agents need full conversational context.
- Typed fields when you want a clean contract and cheap prompts.
- Blackboard when the work is iterative refinement.

## `07_hierarchical_agents.py`

**What it does** — a multi-level org chart using subgraphs:
- Three department `StateGraph`s — research (parallel web-researcher + paper-reviewer fanning into a research lead), content (writer → editor), analysis (data analyst → strategy advisor).
- Each is `.compile()`d and added as a single node in a parent graph.
- A CEO supervisor routes with `with_structured_output(DepartmentRoute)`.
- All levels share one `TeamState` with `add_messages`.

**What it's for**
- Scaling past ~5 agents — a flat supervisor's routing prompt degrades, so group agents into teams and route at two levels.

## Choosing a pattern
- Independent subtasks, need speed → parallel fan-out
- Dynamic order, agents may repeat → supervisor
- Many agents / distinct domains → hierarchical subgraphs
- Single user request that belongs to one specialist → handoffs
- Output quality must be iterated on → blackboard / drafter-critic
