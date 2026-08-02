# Multi-Agent Architecture Overview

## Why multiple agents

One agent with twenty tools and a 2000-word system prompt degrades: it picks wrong tools, ignores instructions, and is impossible to debug. Splitting into specialists with narrow prompts and small toolsets improves reliability, lets you tune each role independently, and makes traces readable. The cost is more LLM calls, more latency, and coordination overhead — so don't reach for it until a single agent visibly fails.

In LangGraph an "agent" is just a node (or a subgraph) with its own prompt and tools. All the patterns below are variations on how control and information flow between those nodes.

## Patterns

### Tool-calling agent (the baseline)
`llm.bind_tools([...])` + `ToolNode` + a conditional edge looping `agent → tools → agent` until no more tool calls. Everything else composes agents built this way.

- **Use when**: a single role, a handful of tools.
- → [`04_multi_agent_systems/01_tool_calling_agent.py`](../04_multi_agent_systems/01_tool_calling_agent.py)

### Supervisor
A coordinator node decides which specialist runs next via `with_structured_output(RouteDecision)` returning a `Literal[...]` plus reasoning; every specialist edges back to the supervisor; a `FINISH` decision routes to a finalize node.

- **Strengths**: order isn't fixed, agents can be revisited (writer → critic → writer), one place to change orchestration policy.
- **Costs**: an extra LLM call between every step; the routing prompt degrades past ~5–7 specialists.
- **Use when**: workflow order depends on intermediate results.
- → [`04_multi_agent_systems/02_supervisor_agent.py`](../04_multi_agent_systems/02_supervisor_agent.py), [`03_multi_agent.py`](../04_multi_agent_systems/03_multi_agent.py)

### Hierarchical (supervisor of supervisors)
Group specialists into department subgraphs, compile each, add them as single nodes in a parent graph, and route at the top level. Research/content/analysis teams under a CEO node in this repo.

- **Strengths**: each router only chooses among 3–4 options; teams are independently testable; departments can have internal parallelism.
- **Costs**: two routing hops of latency; shared state schema must satisfy every level.
- **Use when**: more agents than one supervisor can reason about, or clearly separable domains.
- → [`04_multi_agent_systems/07_hierarchical_agents.py`](../04_multi_agent_systems/07_hierarchical_agents.py)

### Parallel (fan-out / fan-in)
Multiple edges from `START` (or from one node) to independent agents, all edging into a synthesizer. Static fan-out is just edges; **dynamic** fan-out uses the Send API (`[Send("worker", sub_state) for ...]`) when the branch count is runtime-determined. Results merge through reducers, so each agent must write a distinct field or an `operator.add` list.

- **Strengths**: wall-clock latency of the slowest branch, not the sum; genuinely distinct perspectives.
- **Costs**: N× tokens; no branch can see another's output; needs a synthesis step.
- **Use when**: subtasks are independent — multi-source research, map-reduce summarization, multi-angle analysis.
- → [`04_multi_agent_systems/04_parallel_agents.py`](../04_multi_agent_systems/04_parallel_agents.py), [`projects/03_multi_agent_research_system.py`](../projects/03_multi_agent_research_system.py)

### Handoffs
A triage agent classifies the request and transfers control *plus context* to one specialist — `handoff_to`, `reason`, and a `context_summary` written into state, then a conditional edge dispatches. Unlike the supervisor pattern, control doesn't come back.

- **Strengths**: cheap (one routing call, one specialist call); mirrors real customer-service org structure.
- **Costs**: no re-routing if triage guessed wrong; one-shot by design.
- **Use when**: a request belongs to exactly one domain — sales vs support vs billing.
- → [`04_multi_agent_systems/05_agent_handoffs.py`](../04_multi_agent_systems/05_agent_handoffs.py)

### Blackboard / drafter-critic
Agents read and write a shared workspace (`drafts` and `critiques`, both `operator.add` lists) and loop until a structured `ApprovalDecision` approves or an iteration cap fires.

- **Strengths**: output quality improves measurably per round; the critique history is visible and auditable.
- **Costs**: 2 LLM calls per iteration; requires a hard iteration cap or it never terminates.
- **Use when**: quality matters more than latency — content, reports, generated code.
- → [`04_multi_agent_systems/06_agent_communication.py`](../04_multi_agent_systems/06_agent_communication.py)

## Communication mechanisms

Independent of topology, agents share information three ways:

| Mechanism | How | Trade-off |
|---|---|---|
| **Message passing** | `Annotated[list[BaseMessage], add_messages]`; each agent appends a labelled `AIMessage` and reads the whole transcript | Full context, natural for LLMs; prompt grows every hop |
| **Shared typed fields** | Each agent writes its own state key (`raw_data`, `analysis`, `confidence_score`) | Clean contract, cheap prompts, easy to assert on; agents lose conversational nuance |
| **Blackboard** | Shared accumulating workspace both read and written by several agents, usually in a loop | Best for iterative refinement; needs termination logic |

All three are demonstrated side by side in [`04_multi_agent_systems/06_agent_communication.py`](../04_multi_agent_systems/06_agent_communication.py).

## Picking a pattern

1. One role, few tools → **tool-calling agent**
2. Request belongs to exactly one specialist → **handoffs**
3. Independent subtasks, want speed → **parallel fan-out** (Send API if count is dynamic)
4. Order depends on results, agents may repeat → **supervisor**
5. Too many agents for one supervisor → **hierarchical subgraphs**
6. Output must be iterated to quality → **blackboard / drafter-critic**

The Section 4 capstone combines 3, 4 and 6: supervisor plans queries → Send-API parallel search agents → analyst → writer → quality gate looping back to the writer. → [`projects/03_multi_agent_research_system.py`](../projects/03_multi_agent_research_system.py)

## Practical notes

- Always cap iterations on any loop, and force-approve past the cap.
- Use `with_structured_output` for every routing decision — free-text routing is unreliable.
- Name your `AIMessage`s (`name="critic"`) so traces and downstream filters can tell agents apart.
- Watch token growth: `add_messages` state means every agent pays for every prior agent's output.
