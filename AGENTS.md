# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Personal working repo for the Udemy course "Production AI Agents" (LangChain / LangGraph / LangSmith), taught progressively from fundamentals to production patterns. It is a learning/demo codebase, not a shipped application — code quality bar and structure should match that (self-contained demo scripts, not a library).

## Commands

```bash
uv sync                                          # install/sync dependencies (Python 3.12+, uv-managed)
cp .env.example .env                             # then fill in OPENAI_API_KEY / ANTHROPIC_API_KEY (+ optional LANGSMITH_*)
uv run check_api_connection.py                   # verify env setup, prints versions, pings OpenAI + Anthropic
uv run <path>/<script>.py                        # run any demo script — MUST be run from repo root
uv run ruff check .                              # lint (ruff is the only configured dev tool)
uv run ruff format .                             # format
```

There is no pytest suite wired up (`05_production_patterns/05_testing_patterns.py` is a demo script about testing patterns, not actual tests, despite `pytest` being a dependency).

Run scripts from the repo root — several use paths relative to it (e.g. `./assets/sample_docs/langchain_demo.pdf`) and write artifacts (`chroma_db/`, `research_db/`, `*.png`) into the current working directory.

## Code layout and conventions

- **Demo-per-function**: each script has an `if __name__ == "__main__":` block at the bottom where most demo calls are commented out — uncomment the one you want to run before executing.
- **Numeric prefixes = suggested reading/build order** within a topic folder (`01_`, `02_`, …), foundations first. Each topic folder has its own README explaining where the order is a real code dependency vs. just increasing difficulty — check it before assuming independence between files.
- **Exception: `projects/`** — numeric prefixes there are course section numbers, not a difficulty/dependency order.
- Every script calls `load_dotenv()` at import time, so `.env` keys are picked up automatically; never hardcode keys.

## Structure

- `01_langchain_fundamentals/` — LCEL, runnables, prompts, models, output parsers
- `03_langgraph_state_and_control_flow/` — StateGraph, routing, cycles, checkpointing, human-in-the-loop, error handling
- `02_rag_and_memory/` — loaders, splitters, embeddings, vector stores, RAG, conversation memory
- `04_multi_agent_systems/` — tool-calling agents, supervisor, hierarchy, parallel agents, handoffs, communication
- `05_production_patterns/` — LangSmith setup, monitoring, cost optimization, security patterns, testing/eval patterns
- `projects/` — section capstone applications (numbered by course section, not difficulty)
- `assets/graphs/` — exported LangGraph topology PNGs; `assets/sample_docs/` — sample ingestion files
- `docs/` — cross-cutting conceptual primers, meant to be read when returning to the repo cold:
  - `docs/langchain-ecosystem.md` — how LangChain/LangGraph/LangSmith fit together — **read this first** when orienting
  - `docs/langchain-overview.md`, `docs/langgraph-overview.md`, `docs/langsmith-overview.md`, `docs/multi-agent-overview.md`, `docs/rag-overview.md`

Note the numbering mismatch between `01_langchain_fundamentals`, `03_langgraph_state_and_control_flow`, and `02_rag_and_memory` on disk — LangGraph control flow is folder `03` but is the second topic in reading order (see README.md's own folder-guide ordering).
