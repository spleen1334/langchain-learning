"""
Prompt Templates and Messages in LangChain V.1
"""

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_basic_templates():
    """Basic ChatPromptTemplate usage."""

    # Simple template
    simple = ChatPromptTemplate.from_template("Translate '{text}' to {language}")

    messages = simple.format_messages(text="Hello, world!", language="French")
    print("Simple template:")
    print(f"  {messages}")

    # Multi-message template
    multi = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a translator. Be concise."),
            ("human", "Translate '{text}' to {language}"),
        ]
    )

    messages = multi.format_messages(text="Good morning", language="Japanese")
    print("\nMulti-message template:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content}")


def demo_message_types():
    """Working with different message types."""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build conversation manually
    messages = [
        SystemMessage(content="You are a math tutor. Be brief."),
        HumanMessage(content="What's 5 * 5?"),
        # Hand-written AIMessage: you can fabricate the model's "previous" answer to
        # steer it, the API can't tell it didn't actually generate this turn.
        AIMessage(content="25"),
        HumanMessage(content="And if I add 10?"),
    ]

    response = model.invoke(messages)
    print(f"Conversation result: {response.content}")


def demo_messages_placeholder():
    """Use MessagesPlaceholder for dynamic conversation history."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            # Reserves a slot for a *list* of Message objects rather than a string.
            # Injected verbatim at format time, so the model sees real prior turns
            # with their roles intact — this is how chat memory is wired into a prompt.
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Simulate conversation history
    history = [
        HumanMessage(content="My name is Paulo"),
        AIMessage(content="Nice to meet you, Paulo!"),
    ]

    messages = prompt.format_messages(history=history, question="What's my name?")

    print("With history placeholder:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content[:50]}...")

    # Execute
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = prompt | model
    response = chain.invoke({"history": history, "question": "What's my name?"})
    print(f"\nResponse: {response.content}")


def demo_few_shot():
    """Few-shot prompting with examples."""

    # Define examples
    examples = [
        {"word": "happy", "opposite": "sad"},
        {"word": "fast", "opposite": "slow"},
        {"word": "big", "opposite": "small"},
    ]

    # Template for each example
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "What's the opposite of '{word}'?"),
            ("ai", "The opposite of '{word}' is '{opposite}'."),
        ]
    )

    # {word} appears both in the examples and in the final human message, but the
    # few-shot examples are already fully substituted from `examples`, so only the
    # trailing "{word}" stays an open variable at invoke time.
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # Final prompt
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You give the opposite of words. Follow the examples."),
            few_shot,
            ("human", "What's the opposite of '{word}'?"),
        ]
    )

    # Test
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = final_prompt | model

    response = chain.invoke({"word": "bright"})
    print(f"Few-shot result: {response.content}")


def demo_prompt_composition():
    """Compose prompts from reusable parts."""

    # Reusable system prompt
    persona = ChatPromptTemplate.from_messages(
        [("system", "You are a {role}. Your tone is {tone}.")]
    )

    # Reusable task prompt
    task = ChatPromptTemplate.from_messages([("human", "{task}")])

    # Concatenating templates merges their message lists AND their input variables,
    # so one chain can be re-pointed at a new persona purely through invoke() args.
    full_prompt = persona + task

    # Test different combinations
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = full_prompt | model

    # As a pirate
    response = chain.invoke(
        {
            "role": "pirate captain",
            "tone": "adventurous",
            "task": "Tell me about your ship",
        }
    )
    print(f"Pirate: {response.content[:100]}...")

    # As a scientist
    response = chain.invoke(
        {
            "role": "scientist",
            "tone": "precise and academic",
            "task": "Explain photosynthesis",
        }
    )
    print(f"\nScientist: {response.content[:100]}...")


if __name__ == "__main__":
    print("=" * 50)
    print("Demo 1: Basic Templates")
    print("=" * 50)
    demo_basic_templates()

    print("\n" + "=" * 50)
    print("Demo 2: Message Types")
    print("=" * 50)
    demo_message_types()

    print("\n" + "=" * 50)
    print("Demo 3: MessagesPlaceholder")
    print("=" * 50)
    demo_messages_placeholder()

    print("\n" + "=" * 50)
    print("Demo 4: Few-Shot")
    print("=" * 50)
    demo_few_shot()

    print("\n" + "=" * 50)
    print("Demo 5: Prompt Composition")
    print("=" * 50)
    demo_prompt_composition()

    # print("\n" + "=" * 50)
    # print("Exercise: Prompt Library")
    # print("=" * 50)
    # exercise_prompt_library()
