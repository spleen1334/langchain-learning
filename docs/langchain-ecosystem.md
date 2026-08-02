# The LangChain Ecosystem — Quick Reference

**First, a correction to the common "five tools" mental model.** It's easy to come away thinking this ecosystem is five separate products: LangChain, LangGraph, LangSmith, *a monitoring tool*, and *an orchestration tool*. It isn't. There are **three tools** — LangChain, LangGraph, LangSmith — plus two **capabilities** those tools provide:

- **Orchestration** is not a separate product. **LangGraph is the orchestration layer.**
- **Monitoring/observability** is not a separate product. **LangSmith is the monitoring tool.**

So: three things you install, five things you talk about. The sections below keep the five headings you'd expect, but each one names the tool that actually provides it.

---

## LangChain

The component library and composition layer. It gives you provider-agnostic wrappers around chat models and embeddings, prompt templates, output parsers, retrievers, and document loaders — and **LCEL**, the `prompt | model | parser` piping syntax built on the `Runnable` interface (`.invoke()`, `.batch()`, `.stream()`, plus `RunnableParallel` / `RunnableBranch` / `RunnablePassthrough`).

Reach for plain LangChain when your flow is essentially a **directed pipeline**: input goes in, passes through a fixed set of steps, answer comes out. RAG chains, extraction pipelines, and classification chains all fit here. What LangChain does *not* give you is state that persists across steps, loops, or branching that can revisit an earlier step — that's the moment you graduate to LangGraph.

→ [langchain-overview.md](langchain-overview.md) · code: [`01_langchain_fundamentals/`](../01_langchain_fundamentals/), [`03_rag_and_memory/`](../03_rag_and_memory/)

## LangGraph

The **orchestration** framework, and the thing that makes an "agent" rather than a chain. You define a typed state object, register nodes (Python functions that read state and return updates), and wire them with edges — including *conditional* edges, which is how routing, retry loops, and self-correcting cycles become part of the topology instead of hidden inside a node.

Its two headline features are **reducers** (`Annotated[list, operator.add]`, `add_messages`) that merge concurrent writes safely, and **checkpointers** (`MemorySaver`, `SqliteSaver`) that persist state per `thread_id` — which in turn unlock multi-turn memory, time-travel debugging, and human-in-the-loop interrupts. Multi-agent systems (supervisor, hierarchical subgraphs, parallel fan-out, handoffs) are all just LangGraph topologies.

→ [langgraph-overview.md](langgraph-overview.md), [multi-agent-overview.md](multi-agent-overview.md) · code: [`02_langgraph_control_flow/`](../02_langgraph_control_flow/), [`04_multi_agent_systems/`](../04_multi_agent_systems/)

## LangSmith

The **observability and evaluation** platform — a hosted service, not a library you build with. Set `LANGSMITH_TRACING=true` and every LangChain/LangGraph invocation is automatically recorded as a run tree with inputs, outputs, latency, token counts and errors; `@traceable` extends the same treatment to arbitrary Python functions that aren't Runnables.

Beyond tracing it does the evaluation half: **datasets** of test cases, **evaluators** (including LLM-as-judge), and `evaluate(..., experiment_prefix=...)` so you can compare prompt v1 against v2 as a scored experiment rather than by eyeballing outputs. It's independent of the other two — it observes them, they don't depend on it.

→ [langsmith-overview.md](langsmith-overview.md) · code: [`05_production_patterns/01_langsmith_setup.py`](../05_production_patterns/01_langsmith_setup.py), [`05_production_patterns/05_testing_patterns.py`](../05_production_patterns/05_testing_patterns.py)

## Monitoring

**Not a separate tool — this is LangSmith's job**, plus whatever you add around it. In practice production monitoring here comes in two layers:

1. **LangSmith** for LLM-specific observability: trace trees, per-run token usage and cost, prompt/response inspection, error runs, and evaluation scores over time. This is the layer you actually debug with, because it shows you *what the model saw and said*.
2. **Conventional app telemetry** for everything your on-call rotation pages on: structured JSON logs, latency percentiles, error rates, cache hit rates, budget counters — emitted into Datadog/CloudWatch/etc. Nothing LangChain-specific about it; you hand-roll it.

This repo demonstrates both layers side by side.

→ code: [`05_production_patterns/02_monitoring.py`](../05_production_patterns/02_monitoring.py), [`05_production_patterns/03_cost_optimization.py`](../05_production_patterns/03_cost_optimization.py)

## Orchestration

**Also not a separate tool — this is LangGraph's job.** "Orchestration" is just the name for deciding *which step runs next, with what state, and how many times*. LangChain's LCEL does a weak, static form of it (a fixed pipe, plus `RunnableBranch` for a one-shot branch). LangGraph does the real thing: cycles, conditional routing, parallel supersteps with merged state, dynamic fan-out via the `Send` API, subgraphs, and interrupt/resume.

A useful rule of thumb for which layer you need:

| Your flow | Use |
|---|---|
| Fixed sequence of steps | LangChain / LCEL |
| Needs branching, loops, or retries in the topology | LangGraph |
| Needs to pause for a human, or resume after a crash | LangGraph + a checkpointer |
| Several specialists coordinating | LangGraph multi-agent patterns |

And regardless of which you pick, LangSmith sits on top watching it.

→ code: [`02_langgraph_control_flow/03_conditional_edges.py`](../02_langgraph_control_flow/03_conditional_edges.py), [`02_langgraph_control_flow/04_cycles_loops.py`](../02_langgraph_control_flow/04_cycles_loops.py), [`04_multi_agent_systems/`](../04_multi_agent_systems/)

---

## One-line summary

**LangChain** builds the pieces → **LangGraph** orchestrates them into a stateful agent → **LangSmith** watches and grades the result.
