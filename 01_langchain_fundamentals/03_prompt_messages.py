
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# # chatprompttemplate
# prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")


# # format and inspect
# messages = prompt.format_messages(adjective="funny", topic="chickens")

# print(messages)

# Multi-message template: {placeholders} in ANY message (including the system one)
# are filled by the single format_messages() call below.
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

# print(messages)

# model = init_chat_model(model="gpt-4o-mini", temperature=0)
# response = model.invoke(messages)
# print(response.content)

# Message Types:
# from langchain_core.messages import (
#     AIMessage,
#     ChatMessage,
#     HumanMessage,
#     SystemMessage,
#     ToolMessage,
# )

# messages = [
#     HumanMessage(content="Hello!"),
#     AIMessage(content="Hi there! How can I assist you today?"),
#     SystemMessage(content="This is a system message."),
#     ToolMessage(content="Tool executed successfully.", tool_call_id="call_123"),
#     ChatMessage(content="This is a general chat message."),
# ]


# Fewshot example
from langchain_core.prompts import FewShotChatMessagePromptTemplate

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

# Expands each example dict into a human/ai message pair, so the model sees the
# examples as a fake prior conversation rather than as instructions in the prompt text.
fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Give the opposite of each word."),
        # A prompt template can be nested inside another one; it is spliced in as messages.
        fewshot_prompt,
        ("human", "{input}"),
    ]
)


model = init_chat_model(model="gpt-4o-mini", temperature=0)
response = model.invoke(final_prompt.format_messages(input="happy"))

print(response.content)


# Reusable components
system_prompt = ChatPromptTemplate.from_messages([("system", "You are a {role}.")])

user_prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

# `+` on prompt templates concatenates their message lists and unions their
# input variables — handy for reusing a shared system prompt across several chains.
full_prompt = system_prompt + user_prompt


fin = full_prompt.format_messages(role="helpful assistant", question="What is AI?")

print(fin)
