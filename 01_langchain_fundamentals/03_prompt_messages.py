from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

load_dotenv()


def demo_basic_template():
    """ChatPromptTemplate.from_template: single-message prompt filled via format_messages()."""
    prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")
    messages = prompt.format_messages(adjective="funny", topic="chickens")
    print(messages)


def demo_multi_message_template():
    """{placeholders} in ANY message (including the system one) are filled by a
    single format_messages() call."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "Translate the following text: {text}"),
        ]
    )

    messages = prompt.format_messages(
        input_language="English", output_language="French", text="I love programming."
    )
    print(messages)
    return messages


def demo_model_invocation(messages):
    """Send formatted messages to a chat model and print its response."""
    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    response = model.invoke(messages)
    print(response.content)


def demo_message_types():
    """The core message types LangChain uses to represent a conversation."""
    messages = [
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi there! How can I assist you today?"),
        SystemMessage(content="This is a system message."),
        ToolMessage(content="Tool executed successfully.", tool_call_id="call_123"),
        ChatMessage(role="user", content="This is a general chat message."),
    ]
    print(messages)


def demo_fewshot_prompt():
    """Expand example dicts into human/ai message pairs so the model sees the
    examples as a fake prior conversation rather than as instructions in the
    prompt text."""
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    fewshot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # A prompt template can be nested inside another one; it is spliced in as messages.
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give the opposite of each word."),
            fewshot_prompt,
            ("human", "{input}"),
        ]
    )

    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    response = model.invoke(final_prompt.format_messages(input="happy"))
    print(response.content)


def demo_reusable_prompt_components():
    """`+` on prompt templates concatenates their message lists and unions
    their input variables — handy for reusing a shared system prompt across
    several chains."""
    system_prompt = ChatPromptTemplate.from_messages([("system", "You are a {role}.")])
    user_prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

    full_prompt = system_prompt + user_prompt

    messages = full_prompt.format_messages(role="helpful assistant", question="What is AI?")
    print(messages)


if __name__ == "__main__":
    # demo_basic_template()
    demo_fewshot_prompt()
    demo_reusable_prompt_components()
