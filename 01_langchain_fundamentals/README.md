# 01 — LangChain Fundamentals

> **Numeric prefixes = suggested reading order** (increasing conceptual difficulty, not a code dependency — every file runs standalone). `03`→`04` and `06`→`07` are rough-draft → polished-lesson pairs on the same topic.

Core LangChain building blocks: chat models, prompt templates, LCEL composition, and output parsing/structured output. Everything here is provider-agnostic Runnable plumbing that the later sections build on.

## `01_core_concepts.py`
- **What it does**: Introduces LCEL and the `Runnable` interface. Builds `prompt | model | parser` chains with `ChatPromptTemplate.from_template`, `ChatOpenAI`, `StrOutputParser`. Shows the four execution modes — `.invoke()`, `.batch()`, `.stream()` — plus schema introspection via `chain.input_schema.model_json_schema()` / `output_schema`, and `init_chat_model()` as the universal provider-agnostic model constructor.
- **What it's for**: The baseline mental model for every LangChain app — how components compose with `|`, and how to pick between single/batch/streaming execution depending on UX (chat streaming vs bulk offline processing).

## `02_working_with_llms.py`
- **What it does**: Model configuration and multi-provider work. `init_chat_model(model=..., model_provider=..., temperature, streaming, max_retries)` to swap OpenAI/Anthropic without code changes; side-by-side model comparison loop; raw message-object invocation with `SystemMessage`/`HumanMessage` and manual multi-turn by appending the `AIMessage` back into the list.
- **What it's for**: Vendor-portability and model selection — A/B-ing gpt-4o-mini vs gpt-4o vs Claude on the same prompt, and configuring timeouts/retries before production.

## `03_prompt_messages.py`
- **What it does**: Scratchpad on prompts/messages. `ChatPromptTemplate.from_messages` with system+human roles, `.format_messages()` to inspect the rendered messages, the message-type taxonomy (`Human/AI/System/Tool/ChatMessage`), `FewShotChatMessagePromptTemplate` for example-driven prompting, and prompt composition via the `+` operator on two templates.
- **What it's for**: Understanding what actually gets sent to the model, and assembling prompts from reusable persona/task fragments.

## `04_prompt_templates_all.py`
- **What it does**: The polished version of the above, demo-by-demo: basic vs multi-message templates, hand-built conversations from message objects, `MessagesPlaceholder(variable_name="history")` for injecting chat history, few-shot with a per-example `ChatPromptTemplate`, and composing `persona + task` templates parameterized by `{role}`/`{tone}`.
- **What it's for**: Prompt libraries — one template set driving many personas, plus the `MessagesPlaceholder` pattern that every memory-backed chatbot needs.

## `05_chains_v1.py`
- **What it does**: LCEL composition patterns beyond a straight pipe. `RunnableParallel` to fan out summarize/keywords/sentiment on one input; `RunnablePassthrough` + `RunnableLambda` to inject retrieved context (fake retriever standing in for RAG); `RunnableBranch` for classifier-driven routing; and debugging via `chain.with_config(run_name=...)` plus `RunnableLambda` logging taps between steps.
- **What it's for**: Building non-linear chains — multi-aspect analysis in one round trip, intent routing to specialized prompts, and inspecting intermediate values when a chain misbehaves.

## `06_output_parsers_demo.py`
- **What it does**: Quick tour of `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser(pydantic_object=...)` with `.get_format_instructions()` wired in via `prompt.partial(...)`, and `llm.with_structured_output(Model)`.
- **What it's for**: Getting machine-usable data out of an LLM instead of prose — the difference between parsing text after the fact and constraining the model up front.

## `07_output_parsers_final.py`
- **What it does**: The full structured-output lesson: str/JSON/Pydantic parsers, then `with_structured_output()` on a `TaskExtraction` schema (with `Optional` fields for maybe-absent data), and nested schemas (`Company` containing `Address` and `List[str]`). Ends with a movie-extraction exercise using field constraints (`ge=1, le=10`).
- **What it's for**: Extraction pipelines — turning unstructured text (tickets, reviews, emails) into validated typed objects your code can branch on.
