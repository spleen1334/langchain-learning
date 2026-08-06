# LangChain Overview

## What it is

LangChain is a composition framework for LLM applications. It gives you:

- **A uniform interface** over model providers.
- **Composable primitives** — prompts, models, parsers, retrievers, tools.
- **An operator (`|`)** for wiring them together.

It is *not* a model, an agent runtime, or a hosting platform; it's the plumbing layer.

Package layout as of v1:

- `langchain-core` — abstractions, no integrations
- `langchain` — the meta package
- `langchain-openai` / `langchain-anthropic` / `langchain-ollama` / `langchain-chroma` — provider packages
- `langchain-community` — the long tail: loaders, retrievers
- `langchain-classic` — legacy retrievers/storage kept for compat; this repo imports `MultiQueryRetriever`, `ContextualCompressionRetriever`, `EnsembleRetriever` from there
- `langchain-text-splitters`

## Core abstractions

### Runnable + LCEL
Everything implements the `Runnable` interface, so everything gets the same methods for free:

- **Execution** — `.invoke()`, `.batch()`, `.stream()`, `.ainvoke()`
- **Configuration** — `.with_config()`
- **Introspection** — `.input_schema` / `.output_schema`

LCEL (LangChain Expression Language) is the `|` operator composing Runnables into a new Runnable.

```python
chain = prompt | model | parser
```

> [!IMPORTANT]
> Pipe order is not arbitrary — each `Runnable`'s output must match the next one's expected input.
> `prompt` emits a `PromptValue`, `model` consumes a `PromptValue` and emits an `AIMessage`, `parser`
> consumes that message and emits `str`. Reordering (e.g. `model | prompt | parser`) breaks the chain
> at runtime because the types no longer line up.

Composition helpers beyond the pipe:
- `RunnableParallel(a=..., b=...)` — run branches concurrently, return a dict
- `RunnablePassthrough()` — forward the input untouched (used to thread the raw question past a retriever)
- `RunnableLambda(fn)` — lift any Python function into the chain
- `RunnableBranch((cond, chain), default)` — conditional dispatch

→ [`01_langchain_fundamentals/01_core_concepts.py`](../01_langchain_fundamentals/01_core_concepts.py), [`05_chains_v1.py`](../01_langchain_fundamentals/05_chains_v1.py)

### Chat models
- **Provider-agnostic constructor** — `init_chat_model("gpt-4o-mini", model_provider="openai", temperature=..., max_retries=..., streaming=...)`
- **Direct constructors** — `ChatOpenAI` / `ChatAnthropic`
- **Config that matters in production** — `temperature`, `max_tokens`, `timeout`, `max_retries`

→ [`01_langchain_fundamentals/02_working_with_llms.py`](../01_langchain_fundamentals/02_working_with_llms.py), [`check_api_connection.py`](../check_api_connection.py)

### Messages
- **The types** — `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`, `ChatMessage`.
- **A conversation** is just a list of these.
- **Multi-turn** means appending the model's `AIMessage` back before the next `HumanMessage`.
- **Tool calls** live on `AIMessage.tool_calls`; results come back as `ToolMessage`.

→ [`01_langchain_fundamentals/03_prompt_messages.py`](../01_langchain_fundamentals/03_prompt_messages.py)

### Prompt templates
Two constructors:
- `ChatPromptTemplate.from_template(...)` — single-message
- `.from_messages([("system", ...), ("human", ...)])` — role-structured

Key features:
- `.format_messages(**vars)` to see exactly what gets sent
- `.partial(k=v)` to pre-bind a variable (typically parser format instructions)
- `MessagesPlaceholder("history")` to splice in a list of messages at runtime — the hook every memory implementation uses
- `FewShotChatMessagePromptTemplate(example_prompt=..., examples=[...])` for example-driven prompting
- `template_a + template_b` to compose reusable persona/task fragments

→ [`01_langchain_fundamentals/04_prompt_templates_all.py`](../01_langchain_fundamentals/04_prompt_templates_all.py)

### Output parsers and structured output
Two eras, both useful:
- **Parsers** run *after* generation: `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser(pydantic_object=Model)` (pair with `.get_format_instructions()` in the prompt).
- **`model.with_structured_output(Model)`** constrains generation up front using the provider's native tool/JSON-schema support. Returns a validated Pydantic instance — no format instructions needed. This is the default choice now, and it powers routing decisions throughout sections 4 and 5.

→ [`01_langchain_fundamentals/07_output_parsers_final.py`](../01_langchain_fundamentals/07_output_parsers_final.py)

### Retrieval
- **The unit** — `Document(page_content, metadata)`.
- **The chain of custody** — loaders produce Documents, splitters chop them, embeddings vectorize them, vector stores index them.
- **The handoff into LCEL** — `vectorstore.as_retriever(...)` exposes a Runnable you can drop straight into a chain.

See [rag-overview.md](rag-overview.md).

### Tools
- **`@tool`** on a typed, docstring'd Python function generates the schema.
- **`model.bind_tools([...])`** exposes them to the model.
- **Execution is your job** — or `ToolNode`'s, in LangGraph.

→ [`04_multi_agent_systems/01_tool_calling_agent.py`](../04_multi_agent_systems/01_tool_calling_agent.py)

## Where each concept lives in this repo

| Concept | File |
|---|---|
| LCEL, invoke/batch/stream, schemas | `01_langchain_fundamentals/01_core_concepts.py` |
| Providers, `init_chat_model`, model comparison | `01_langchain_fundamentals/02_working_with_llms.py` |
| Message types, few-shot, prompt composition | `01_langchain_fundamentals/03_prompt_messages.py`, `04_prompt_templates_all.py` |
| Parallel / passthrough / branch / debugging | `01_langchain_fundamentals/05_chains_v1.py` |
| Parsers + `with_structured_output` | `01_langchain_fundamentals/06_output_parsers_demo.py`, `07_output_parsers_final.py` |
| Memory (`RunnableWithMessageHistory`, trimming, summary) | `02_rag_and_memory/08_conversation_memory.py` |
| Full RAG chain | `02_rag_and_memory/06_rag_pipeline.py` |
| Capstone using all of it | `projects/01_smart_bot_section1.py`, `projects/02_research_assistant.py` |

## When LangChain stops being enough

A chain is a DAG that runs once. Switch to LangGraph the moment you need any of:

- Loops
- Conditional revisiting of an earlier step
- Persisted state across turns
- A human pause in the middle

See [langgraph-overview.md](langgraph-overview.md).
