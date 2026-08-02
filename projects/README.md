# Projects — Section Capstones

> **Numeric prefixes = course section order**, not difficulty — these are the end-of-section capstones (Section 1, 2, 4). Each is standalone; read the matching topic folder first.

Each script is the end-of-section project, combining the concepts taught in that section into one runnable application. Read these to see how the isolated demos fit together.

## `01_smart_bot_section1.py` — Section 1: Smart Q&A Bot

**What it does**
- A `SmartQABot` class wrapping `ChatOpenAI(...).with_structured_output(QAResponse)` behind a system-prompt `ChatPromptTemplate`.
- `QAResponse` carries answer, confidence, reasoning, follow-up questions and a `sources_needed` flag.
- `ask()` is `@traceable`-decorated and catches exceptions into a valid low-confidence `QAResponse` rather than raising.
- `ask_batch()` uses `chain.batch()` for parallel questions.
- LangSmith tracing self-configures if `LANGSMITH_API_KEY` is present.

**What it's for**
- The Section 1 payoff — prompts + structured output + graceful degradation + tracing is already a shippable service endpoint.

## `02_research_assistant.py` — Section 2: AI Research Assistant

**What it does** — an `AIResearchAssistant` class owning the whole RAG stack:
- `OpenAIEmbeddings`, `RecursiveCharacterTextSplitter`, a persistent `Chroma` collection, and per-session `InMemoryChatMessageHistory`.
- Ingestion (`add_documents`/`add_text`/`add_texts`) tags source metadata and an `indexed_at` timestamp.
- `_build_retriever(use_advanced=)` toggles plain similarity vs `MultiQueryRetriever`.
- `ask()` returns cited prose; `ask_structured()` returns a `ResearchResponse` (answer, confidence, sources, key quotes, follow-ups) via `with_structured_output`.
- `compare_retrievers()` prints basic vs advanced chunk counts and total characters sent to the LLM.

**What it's for**
- A complete document-Q&A product — ingest, retrieve, answer with citations.
- Remembering the conversation across follow-up questions ("how does the second component work?").

## `03_multi_agent_research_system.py` — Section 4: Multi-Agent Research System

**What it does** — a five-node LangGraph research pipeline:
- A supervisor plans 3 search queries.
- `dispatch_searches` fans out via the **Send API** (`Send("search_agent", {...})` returned from a conditional edge), so one agent instance runs per query and results merge into an `Annotated[list[dict], operator.add]` findings blackboard.
- An analyst synthesizes; a report writer drafts markdown.
- A `quality_checker` scores it with `with_structured_output(QualityReview)`, and a `quality_gate` edge loops back to the writer until score ≥ 0.7 or 2 iterations.
- Includes a `stream(stream_mode="updates")` demo and exports the graph PNG.

**What it's for**
- The Section 4 payoff — supervisor + dynamic parallelism + shared blackboard + quality-gated refinement in one system.
- The Send API is the piece worth remembering: fan-out width decided at runtime, not graph-build time.
