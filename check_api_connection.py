from importlib.metadata import version

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Must run before any chat model is constructed: LangChain reads OPENAI_API_KEY /
# ANTHROPIC_API_KEY from os.environ at construction time, not at invoke() time.
load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")

# Which provider to actually test.
# Valid values: "openai" | "anthropic" | "both"
PROVIDER = "openai"

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def test_openai():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    # Printing the whole AIMessage (not .content) so you can see the metadata:
    # token usage, finish reason, model name — useful to confirm the call really hit the API.
    print(f"Response from ChatOpenAI: {response}")


def test_anthropic():
    llm_anthropic = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatAnthropic: {response_anthropic}")


def main():
    # Switch statement over PROVIDER — only the selected branch actually calls out to an API.
    match PROVIDER:
        case "openai":
            test_openai()
        case "anthropic":
            test_anthropic()
        case "both":
            test_openai()
            test_anthropic()
        case _:
            raise ValueError(
                f"Unknown PROVIDER: {PROVIDER!r} (expected 'openai', 'anthropic', or 'both')"
            )

    print("Setup complete!")


if __name__ == "__main__":
    main()
