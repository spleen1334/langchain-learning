# Conversation History vs. LangGraph Checkpointing

Both approaches can make a chatbot remember earlier turns, but they persist state at different architectural levels.

## Short version

- **`RunnableWithMessageHistory`** manages a conversation transcript around a LangChain runnable.
- **A LangGraph checkpointer** saves snapshots of an entire graph's state and execution progress.

| | `RunnableWithMessageHistory` | LangGraph checkpointer |
|---|---|---|
| Primary purpose | Give a runnable previous chat messages | Persist and resume a stateful workflow |
| Stored application data | Chat messages | Every field in the graph state schema, which may include messages |
| Runtime information | None | Pending nodes, checkpoint lineage, node writes, step metadata, and timestamps |
| Conversation key | Usually `session_id` | `thread_id` |
| Save boundary | Before and after a runnable invocation | At graph super-step boundaries and interrupts |
| Supports interrupts and resume | No | Yes |
| Supports state inspection, replay, and branching | No | Yes |

## Message history and its runnable wrapper

`InMemoryChatMessageHistory` is a container for message objects. By itself, it does not load a session, modify a prompt, invoke a model, or save the result.

`RunnableWithMessageHistory` supplies that orchestration. On each invocation it:

1. Uses `session_id` to obtain a `BaseChatMessageHistory` instance.
2. Injects the stored messages into the wrapped runnable.
3. Invokes the runnable with the new input.
4. Appends the new input and output messages to the history.

The backing history can be process-local, such as `InMemoryChatMessageHistory`, or durable, such as `SQLChatMessageHistory`. Changing the backend changes durability, but the stored application data remains a chat transcript.

See [`02_rag_and_memory/08_conversation_memory.py`](../02_rag_and_memory/08_conversation_memory.py).

## LangGraph checkpointing

A checkpointer is integrated with the LangGraph runtime. Given a graph state such as:

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_plan: list[str]
    approval_status: str
    retry_count: int
```

the checkpointer saves all four state fields, not only `messages`. A checkpoint also records execution information such as:

- which nodes are scheduled to execute next;
- the `thread_id` and unique `checkpoint_id`;
- the parent checkpoint;
- step and node-write metadata;
- interrupts, errors, and pending work where applicable;
- the checkpoint creation time.

That additional information lets LangGraph inspect state, pause for human input, resume execution, recover work, replay an earlier step, or branch from an earlier checkpoint.

See [`03_langgraph_state_and_control_flow/05_checkpointing.py`](../03_langgraph_state_and_control_flow/05_checkpointing.py).

## Why they can look equivalent

If a graph's only state field is `messages`, both approaches produce the same visible chatbot behavior: a later turn can see earlier messages.

They are still not equivalent internally. Message history stores the resulting transcript. A checkpointer stores successive workflow snapshots, including the transcript, graph position, and checkpoint metadata.

Therefore, a checkpointer is the LangGraph mechanism for thread-level conversational continuity, but it is broader than a chat-history store.

## Which one to use

Use `RunnableWithMessageHistory` for a simple LangChain pipeline such as:

```text
prompt -> model -> parser
```

Use a LangGraph checkpointer when the application has stateful workflow behavior such as:

```text
state -> model/tool nodes -> routing -> loops -> interrupts
```

For an agent already implemented with LangGraph—or a modern LangChain agent backed by LangGraph—prefer its checkpointer instead of wrapping the whole agent with a second message-history mechanism. Keeping two independent stores for the same conversation can create conflicting sources of truth.

Finally, **in-memory describes durability, not architecture**. Both `InMemoryChatMessageHistory` and `MemorySaver`/`InMemorySaver` lose their contents when the process exits. Production deployments normally replace them with persistent message-history or checkpoint backends.
