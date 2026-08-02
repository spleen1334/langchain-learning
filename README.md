# Production AI Agents — Course Code

Personal working repo for the Udemy course [Production AI Agents](https://www.udemy.com/course/production-ai-agents/) (LangChain / LangGraph / LangSmith), taught progressively from fundamentals to production patterns.

Scripts are demo-per-function with a `if __name__ == "__main__":` block at the bottom where most demos are commented out — uncomment the one you want and run the file.

Files inside each topic folder are prefixed `01_`, `02_`, … reflecting **suggested reading order** — foundations first, building up. Each folder's README explains its ordering and calls out where the sequence is a real code dependency rather than just increasing difficulty. In `projects/` the numbers are course section numbers, not difficulty.

## Structure

```
.
├── 01_langchain_fundamentals/   LCEL, runnables, prompts, models, output parsers
│     01_core_concepts · 02_working_with_llms · 03_prompt_messages · 04_prompt_templates_all
│     05_chains_v1 · 06_output_parsers_demo · 07_output_parsers_final
├── 02_langgraph_control_flow/   StateGraph, routing, cycles, checkpointing, HITL, errors
│     01_langgraph_core · 02_first_graph · 03_conditional_edges · 04_cycles_loops
│     05_checkpointing · 06_human_in_loop · 07_error_handling
├── 03_rag_and_memory/           loaders, splitters, embeddings, vector stores, RAG, memory
│     01_document_loaders · 02_text_splitters · 03_embeddings · 04_embeddings_deep
│     05_vector_stores · 06_rag_pipeline · 07_advanced_rag · 08_conversation_memory
├── 04_multi_agent_systems/      tools, supervisor, hierarchy, parallel, handoffs, comms
│     01_tool_calling_agent · 02_supervisor_agent · 03_multi_agent · 04_parallel_agents
│     05_agent_handoffs · 06_agent_communication · 07_hierarchical_agents
├── 05_production_patterns/      LangSmith, monitoring, cost, security, testing & eval
│     01_langsmith_setup · 02_monitoring · 03_cost_optimization
│     04_security_patterns · 05_testing_patterns
├── projects/                    section capstone applications (numbered by course section)
│     01_smart_bot_section1 · 02_research_assistant · 03_multi_agent_research_system
├── assets/graphs/               exported LangGraph topology PNGs
├── assets/sample_docs/          sample files for ingestion demos (langchain_demo.pdf)
├── docs/                        cross-cutting concept overviews
├── check_api_connection.py                      setup verification (checks both API keys work)
└── pyproject.toml / uv.lock     uv-managed dependencies
```

## Folder guides

- [01 — LangChain Fundamentals](01_langchain_fundamentals/README.md)
- [02 — LangGraph Control Flow](02_langgraph_control_flow/README.md)
- [03 — RAG and Memory](03_rag_and_memory/README.md)
- [04 — Multi-Agent Systems](04_multi_agent_systems/README.md)
- [05 — Production Patterns](05_production_patterns/README.md)
- [Projects (section capstones)](projects/README.md)
- [Graph visualizations](assets/graphs/README.md)

## Concept overviews

Start here when coming back cold — conceptual primers with links into the code.

- [The LangChain ecosystem](docs/langchain-ecosystem.md) — how LangChain, LangGraph and LangSmith fit together (and where "orchestration" and "monitoring" actually live). **Read this first.**
- [LangChain](docs/langchain-overview.md) — runnables, LCEL, prompts, parsers, structured output
- [LangGraph](docs/langgraph-overview.md) — state, reducers, edges, cycles, checkpointing, human-in-the-loop
- [LangSmith](docs/langsmith-overview.md) — tracing, monitoring, cost tracking, datasets & evaluation
- [Multi-agent architectures](docs/multi-agent-overview.md) — supervisor, hierarchical, parallel, handoffs, blackboard
- [RAG](docs/rag-overview.md) — the ingest→retrieve→generate pipeline, advanced retrieval, memory

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # install dependencies from pyproject.toml / uv.lock
```

Copy `.env.example` to `.env` and fill in real values (`.env` is gitignored — never commit it):

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# optional, for section 05 and any @traceable code
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=production-ai-agents
```

Every script calls `load_dotenv()` at import, so the keys are picked up automatically.

Verify the setup:

```bash
uv run check_api_connection.py          # prints library versions and pings OpenAI + Anthropic
```

Then run any demo:

```bash
uv run 02_langgraph_control_flow/05_checkpointing.py
```

Note: a few scripts use paths relative to the repo root (e.g. `./assets/sample_docs/langchain_demo.pdf`) and write artifacts (`chroma_db/`, `research_db/`, `*.png`) into the current working directory — run them from the repo root.
