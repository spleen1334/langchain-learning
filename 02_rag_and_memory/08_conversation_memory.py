"""
Conversation Memory in LangChain
Modern approaches to maintaining conversation context
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()


llm = init_chat_model("gpt-4o-mini")


def demo_basic_memory():
    """Basic conversation memory with RunnableWithMessageHistory."""

    print("=" * 60)
    print("BASIC CONVERSATION MEMORY")
    print("Using RunnableWithMessageHistory (modern approach)")
    print("=" * 60)

    # Prompt with history placeholder
    # MessagesPlaceholder splices in a variable-length list of message objects (the
    # growing history); "{input}" is a plain string slot for exactly one human turn.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Be concise."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    # Session storage
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Wraps the chain so that on each invoke it: loads history for the session, injects
    # it into `history`, runs the chain, then appends BOTH the input and the output to
    # the store automatically. The keys must match the prompt's variable names.
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # session_id is looked up by get_session_history — the LangChain equivalent of
    # LangGraph's thread_id. Omitting it raises rather than silently sharing memory.
    config: RunnableConfig = {"configurable": {"session_id": "user_123"}}

    # Conversation
    messages = [
        "Hi! My name is Anatolij.",
        "I'm learning about LangChain.",
        "What's my name and what am I learning?",
    ]

    print("\nConversation:")
    for msg in messages:
        print(f"\nUser: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI: {response}")

    # Show stored history
    print(f"\n--- Stored History ({len(store['user_123'].messages)} messages) ---")
    for msg in store["user_123"].messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content[:50]}...")


def demo_multi_sessions():

    print("=" * 60)
    print("MULTIPLE CONVERSATION SESSIONS")
    print("Each user gets their own memory")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Remember user details."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # ONE chain object serves both users — isolation comes entirely from the session_id
    # in the config, so a chain can be built once and shared across requests/threads.
    user_a_config: RunnableConfig = {"configurable": {"session_id": "user_a"}}
    user_b_config: RunnableConfig = {"configurable": {"session_id": "user_b"}}

    # User A conversation
    print("\n--- User A ---")
    print("User A: My favorite language is Python")
    resp = chain_with_history.invoke(
        {"input": "My favorite language is Python"}, config=user_a_config
    )
    print(f"AI: {resp}")

    # User B conversation
    print("\n--- User B ---")
    print("User B: I love JavaScript")
    resp = chain_with_history.invoke(
        {"input": "I love JavaScript"}, config=user_b_config
    )
    print(f"AI: {resp}")

    # Ask each user about their preference
    print("\n--- Asking each about their preference ---")

    print("\nUser A: What's my favorite language?")
    resp = chain_with_history.invoke(
        {"input": "What's my favorite language?"}, config=user_a_config
    )
    print(f"AI: {resp}")

    print("\nUser B: What's my favorite language?")
    resp = chain_with_history.invoke(
        {"input": "What's my favorite language?"}, config=user_b_config
    )
    print(f"AI: {resp}")


def demo_message_trimming():
    """Trim messages to fit context window."""

    print("=" * 60)
    print("MESSAGE TRIMMING")
    print("Keep conversation within token limits")
    print("=" * 60)

    # Simulate a long conversation
    messages = [
        SystemMessage(content="You are a helpful coding assistant."),
        HumanMessage(content="What is Python?"),
        AIMessage(
            content="Python is a high-level programming language known for readability and versatility. It's used in web development, data science, AI, and automation."
        ),
        HumanMessage(content="How do I install it?"),
        AIMessage(
            content="You can install Python from python.org or use package managers like apt, brew, or chocolatey. I recommend Python 3.12+ for new projects."
        ),
        HumanMessage(content="What about pip?"),
        AIMessage(
            content="Pip is Python's package installer. It comes with Python 3.4+. Use 'pip install package_name' to install packages. Consider using virtual environments with venv or uv."
        ),
        HumanMessage(content="Can you summarize everything we discussed?"),
    ]

    print(f"\nOriginal: {len(messages)} messages")

    # Trim to last N tokens
    trimmed = trim_messages(
        messages,
        max_tokens=60,
        # "last" keeps the most RECENT messages (drops from the front) — usually what you
        # want, since recent turns matter most. "first" would do the opposite.
        strategy="last",
        # Passing the model itself makes it count tokens with that model's real tokenizer
        # rather than a character-count estimate.
        token_counter=llm,
        include_system=True,  # Always keep system message
        # False = never cut a message in half; drop it whole. Half a message can leave
        # dangling tool calls or nonsense context.
        allow_partial=False,
    )

    print(f"After trimming (max 60 tokens): {len(trimmed)} messages")
    print("\nTrimmed messages:")
    for msg in trimmed:
        role = type(msg).__name__.replace("Message", "")
        print(f"  {role}: {msg.content[:60]}...")


def demo_windowed_memory():
    """Implement sliding window memory manually."""

    print("=" * 60)
    print("WINDOWED MEMORY (Keep Last K)")
    print("Fixed-size conversation window")
    print("=" * 60)

    class WindowedChatHistory(InMemoryChatMessageHistory):
        """Chat history that keeps only last k message pairs."""

        # Declared as a class-level annotated field because the parent is a Pydantic
        # model — a plain __init__ assignment would be rejected.
        k: int = 3  # Pydantic field - number of exchange pairs to keep

        # Overriding the write path (not the read path) means trimming happens once on
        # append rather than on every retrieval, and the store never grows unbounded.
        def add_messages(self, messages):
            super().add_messages(messages)
            # Keep only last k pairs (2k messages: human + ai)
            if len(self.messages) > self.k * 2:
                # Rebinding to a slice (rather than del/pop in place) — the discarded
                # messages are gone for good; this window forgets, it doesn't archive.
                self.messages = self.messages[-(self.k * 2) :]

    store: dict[str, WindowedChatHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = WindowedChatHistory(k=2)
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config: RunnableConfig = {"configurable": {"session_id": "windowed_test"}}

    # Simulate a conversation with more than 2 pairs
    exchanges = [
        "My name is Anatolij",
        "I live in Seattle",
        "I work as an AI engineer",
        "I have 2 cats",
        "What do you remember about me?",
    ]

    print("\nConversation with k=2 window:")
    for i, msg in enumerate(exchanges, 1):
        print(f"\nUser: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI: {response}")

        # Show window state after each exchange so students SEE it sliding
        history = store["windowed_test"].messages
        print(f"  [Window: {len(history)} msgs] ", end="")
        facts_in_memory = [
            m.content[:40] for m in history if isinstance(m, HumanMessage)
        ]
        print(f"Remembers: {facts_in_memory}")

    # Final state - show what survived and what was lost
    print("\n" + "=" * 60)
    print("RESULT: Window only kept last 2 exchanges!")
    print("Lost: name (Anatolij), city (Seattle), AND job (AI engineer)")
    print("Kept: cats + the 'remember' question")
    print(
        "This is the tradeoff: fixed memory = predictable cost, but older context is lost."
    )


def demo_summary_memory():
    """End-to-end summary memory: auto-summarize old messages, keep recent ones verbatim."""

    print("=" * 60)
    print("SUMMARY MEMORY")
    print("Summarize older messages to save tokens")
    print("=" * 60)

    # --- Setup ---
    # Two separate models on purpose: temperature=0 for summarization (facts must not
    # drift as they're re-compressed each round) and 0.7 for the conversational replies.
    summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chat_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # The conversation prompt: summary of old context + recent messages
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant. Be concise.\n\n"
                    "Summary of earlier conversation:\n{summary}"
                ),
            ),
            MessagesPlaceholder(variable_name="recent_messages"),
            ("human", "{input}"),
        ]
    )

    chat_chain = chat_prompt | chat_llm | StrOutputParser()

    # The summarization prompt: compress messages into a running summary
    # Takes the OLD summary plus the newly-evicted messages and folds them together, so
    # the summary is refined incrementally instead of re-summarizing the whole history.
    summarize_prompt = ChatPromptTemplate.from_template(
        "Condense the current summary and new messages into a single updated summary "
        "(2-3 sentences). Preserve all key facts about the user.\n\n"
        "Current summary:\n{current_summary}\n\n"
        "New messages:\n{new_messages}\n\n"
        "Updated summary:"
    )

    summarize_chain = summarize_prompt | summary_llm | StrOutputParser()

    # --- State ---
    running_summary = ""  # starts empty
    recent_messages = []  # full message objects
    MAX_RECENT = 4  # keep last 4 messages (2 exchanges) before summarizing

    # --- Conversation ---
    exchanges = [
        "My name is Anatolij and I'm from Seattle",
        "I work as an AI engineer building RAG systems",
        "I have 2 cats named Luna and Milo",
        "I'm taking a LangChain course on Udemy",
        "What do you know about me? List everything.",
    ]

    print(f"\nConfig: keep last {MAX_RECENT} messages, summarize the rest\n")

    for user_input in exchanges:
        print(f"User: {user_input}")

        # 1. Call the LLM with summary + recent messages + new input
        response = chat_chain.invoke(
            {
                "summary": (
                    running_summary if running_summary else "No prior conversation."
                ),
                "recent_messages": recent_messages,
                "input": user_input,
            }
        )
        print(f"AI: {response}")

        # 2. Add this exchange to recent messages
        recent_messages.append(HumanMessage(content=user_input))
        recent_messages.append(AIMessage(content=response))

        # 3. If recent messages exceed limit, summarize the oldest ones
        if len(recent_messages) > MAX_RECENT:
            # Take the oldest messages that will be summarized away
            # Everything EXCEPT the last MAX_RECENT — those are the messages about to
            # fall out of the verbatim buffer, so they must be folded into the summary
            # BEFORE the buffer is truncated below, or the facts would be lost.
            messages_to_summarize = recent_messages[:-MAX_RECENT]
            formatted = "\n".join(
                f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in messages_to_summarize
            )

            # Update the running summary
            running_summary = summarize_chain.invoke(
                {
                    "current_summary": (
                        running_summary if running_summary else "None yet."
                    ),
                    "new_messages": formatted,
                }
            )

            # Keep only the most recent messages
            recent_messages = recent_messages[-MAX_RECENT:]

            print(
                f"  >>> Summarized! Compressed {len(messages_to_summarize)} old messages"
            )
            print(f"  >>> Summary: {running_summary}")
            print(f"  >>> Recent buffer: {len(recent_messages)} messages")
        print()

    # --- Final state ---
    print("=" * 60)
    print("FINAL MEMORY STATE")
    print("=" * 60)
    print(f"\nRunning summary (compressed old context):\n  {running_summary}")
    print(f"\nRecent messages kept verbatim ({len(recent_messages)}):")
    for msg in recent_messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content[:80]}")
    print("\nKey insight: ALL facts preserved (name, city, job, cats, course)")
    print("But token cost stays bounded -- old messages are compressed, not deleted!")


# Exercise
def exercise_persistent_memory():
    """
    EXERCISE: Build a chatbot with:
    1. Persistent memory (SQLite)
    2. Automatic summarization after 10 messages
    3. User preference tracking

    Hint: Combine RunnableWithMessageHistory with SQLChatMessageHistory
    """

    print("=" * 60)
    print("EXERCISE: Persistent Memory Chatbot")
    print("=" * 60)

    import os

    from langchain_community.chat_message_histories import SQLChatMessageHistory

    # Use SQLite for persistence
    db_path = "./chat_history.db"

    # A NEW SQLChatMessageHistory per call is fine (and intended): it holds no state
    # itself, reading/writing the DB on demand, so nothing lives in process memory.
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return SQLChatMessageHistory(
            session_id=session_id, connection=f"sqlite:///{db_path}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Remember user preferences."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config: RunnableConfig = {"configurable": {"session_id": "persistent_user"}}

    print("\nPersistent memory chatbot:")
    print("(Messages saved to SQLite database)\n")

    # Test conversation
    test_messages = [
        "Remember that I prefer dark mode themes",
        "What theme do I prefer?",
    ]

    for msg in test_messages:
        print(f"User: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI: {response}\n")

    print(f"Database created: {db_path}")
    print("Messages persist across restarts!")

    # Cleanup for demo
    if os.path.exists(db_path):
        os.remove(db_path)


def exercise_persistent_memory_proof():
    """
    EXERCISE: Build a chatbot with:
    1. Persistent memory (SQLite)
    2. Proof that messages survive across separate chain instances
    3. User preference tracking

    Key idea: We create the chain TWICE to simulate two separate program runs.
    The second run reads from the same SQLite DB and recalls what the first run stored.
    """

    print("=" * 60)
    print("EXERCISE: Persistent Memory Chatbot")
    print("=" * 60)

    import os
    import sqlite3

    from langchain_community.chat_message_histories import SQLChatMessageHistory

    db_path = "./chat_history.db"
    connection_string = f"sqlite:///{db_path}"
    session_id = "persistent_user"

    # Clean slate
    if os.path.exists(db_path):
        os.remove(db_path)

    # --- Helper: build a fresh chain (simulates a new program run) ---
    def build_chain():
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        def get_session_history(sid: str) -> BaseChatMessageHistory:
            return SQLChatMessageHistory(
                session_id=sid,
                connection=connection_string,
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Remember user preferences and facts.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        chain = prompt | llm | StrOutputParser()

        return RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    config: RunnableConfig = {"configurable": {"session_id": session_id}}

    # =====================================================
    # RUN 1 -- Store preferences (simulates first session)
    # =====================================================
    print("\n--- RUN 1: Storing preferences ---\n")

    chain_v1 = build_chain()

    run1_messages = [
        "My name is Anatolij. I prefer dark mode themes and Python over JavaScript.",
        "I also like my responses concise -- no fluff.",
    ]

    for msg in run1_messages:
        print(f"User: {msg}")
        response = chain_v1.invoke({"input": msg}, config=config)
        print(f"AI:   {response}\n")

    # Dropping the chain proves the point: if any history were cached on the object,
    # RUN 2 below would fail to recall anything.
    del chain_v1

    # =====================================================
    # PROOF: Inspect the raw SQLite database
    # =====================================================
    print("--- DATABASE PROOF ---\n")
    print(f"Database file exists: {os.path.exists(db_path)}")
    print(f"Database size: {os.path.getsize(db_path)} bytes\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT * FROM message_store ORDER BY rowid")
    rows = cursor.fetchall()
    print(f"Total messages stored in DB: {len(rows)}\n")

    for i, row in enumerate(rows):
        print(
            f"  Row {i + 1}: session={row[0] if len(row) > 0 else 'N/A'}, "
            f"message (first 80 chars): {str(row[1])[:80] if len(row) > 1 else 'N/A'}..."
        )
    conn.close()

    # =====================================================
    # RUN 2 -- Brand new chain, same DB (simulates restart)
    # =====================================================
    print("\n--- RUN 2: Fresh chain, testing recall ---\n")

    chain_v2 = build_chain()

    recall_questions = [
        "What's my name?",
        "What theme do I prefer?",
        "What programming language do I prefer?",
        "How do I like my responses?",
    ]

    for msg in recall_questions:
        print(f"User: {msg}")
        response = chain_v2.invoke({"input": msg}, config=config)
        print(f"AI:   {response}\n")

    del chain_v2

    # =====================================================
    # FINAL: Show total messages accumulated
    # =====================================================
    print("--- FINAL DATABASE STATE ---\n")
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM message_store")
    count = cursor.fetchone()[0]
    conn.close()

    print(f"Total messages in DB after both runs: {count}")
    print("Key insight: The second chain had ZERO in-memory history.")
    print("Everything was loaded from SQLite -- true persistence!")

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    # demo_basic_memory()
    # demo_multi_sessions()
    # demo_message_trimming()
    # demo_windowed_memory()
    demo_summary_memory()
    # exercise_persistent_memory()
    # exercise_persistent_memory_proof()
