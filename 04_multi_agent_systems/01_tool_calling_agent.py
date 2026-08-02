"""
Tool-Calling Agents with LangGraph
Building agents that can use tools
"""

from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


@tool
def calculate(expression: str) -> str:
    # The @tool decorator turns this into a schema for the model: the function NAME,
    # the type hints, and THIS DOCSTRING are all sent to the LLM as the tool spec.
    # The docstring is therefore prompt engineering, not just documentation.
    """Calculate a mathematical expression. Example: calculate('2 + 2')"""
    try:
        result = eval(expression)  # Note: In production, use a safe math parser
        return f"The result of {expression} is {result}"
    # Errors are RETURNED as a string, not raised: the text becomes a ToolMessage the
    # model can read and react to (retry, apologise). A raised exception would kill the graph.
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        "new york": "72°F, Sunny",
        "london": "58°F, Cloudy",
        "tokyo": "68°F, Clear",
        "paris": "65°F, Partly Cloudy",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather data not available for {city}"


@tool
def search_web(query: str) -> str:
    """Simulate a web search for a query."""
    # Simulated search results
    search_results = {
        "python programming": "Python is a high-level programming language known for its readability and versatility.",
        "latest news": "Today's top news: AI continues to advance, impacting various industries worldwide.",
        "best restaurants in new york": "Top restaurants in New York include Le Bernardin, Per Se, and Eleven Madison Park.",
    }
    query_lower = query.lower()
    if query_lower in search_results:
        return f"Search results for '{query}': {search_results[query_lower]}"
    return f"No search results found for '{query}'"


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def create_tool_agent():
    """Create a basic tool-calling agent."""

    tools = [calculate, get_weather, search_web]
    # bind_tools attaches the tool schemas to every request. The model never EXECUTES
    # anything — it only emits a `tool_calls` list on the AIMessage; ToolNode runs them.
    llm_with_tools = llm.bind_tools(tools)  # this is the secret!

    def agent_node(state: AgentState) -> str:
        # Generate a response using the LLM with tool access
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Check if we should continue to tools or end."""
        last_message = state["messages"][-1]

        # The ONLY termination signal in an agent loop: an AIMessage with no tool_calls
        # means the model produced a final answer instead of requesting another tool.
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        return "tools"

    # Prebuilt node that reads tool_calls off the last AIMessage, runs the matching
    # functions (in parallel if there are several), and appends one ToolMessage per call
    # with the matching tool_call_id — providers reject tool results without that id.
    tool_node = ToolNode(tools)

    # create graph
    graph = StateGraph(AgentState)

    # add nodes and edges
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )

    # agent -> tools -> agent is the ReAct loop: the model sees the tool results and
    # decides whether to call more tools or answer. Unconditional edge, because after
    # running tools the model must ALWAYS get a chance to interpret them.
    graph.add_edge("tools", "agent")  # loop back after tool execution

    return graph.compile()


def demo_tool_agent():
    """Demo the tool-calling agent."""

    agent = create_tool_agent()

    queries = [
        "What's 25 * 17?",
        "What's the weather in Tokyo?",
        "What's 100 / 4 and what's the weather in London?",
    ]

    print("Tool-Calling Agent Demo:\n")

    for query in queries:
        print(f"Query: {query}")

        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        # Get final response
        final_message = result["messages"][-1]
        print(f"Response: {final_message.content}")
        # Message count reveals the loop: 1 human + 1 AI tool request + N tool results
        # + 1 final AI answer. The third query needs two tools, hence more messages.
        print(f"Total messages: {len(result['messages'])}")
        print("-" * 40)


def demo_tool_execution_trace():
    """Show detailed tool execution trace."""

    agent = create_tool_agent()

    print("\nTool Execution Trace:\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="Calculate 15% of 250 and check weather in Paris")
            ]
        }
    )

    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        print(f"\n[{i}] {msg_type}:")

        if isinstance(msg, HumanMessage):
            print(f"  Content: {msg.content}")

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"  Tool calls: {len(msg.tool_calls)}")
                for tc in msg.tool_calls:
                    print(f"    - {tc['name']}({tc['args']})")
            else:
                print(f"  Content: {msg.content}")

        elif isinstance(msg, ToolMessage):
            print(f"  Tool: {msg.name}")
            print(f"  Result: {msg.content}")


@tool
def divide(a: float, b: float) -> str:
    """Divide two numbers."""
    # Guarding inside the tool keeps the failure inside the conversation: the model gets
    # this string back and can explain it, rather than the graph raising ZeroDivisionError.
    if b == 0:
        return "Error: Division by zero"
    result = a / b
    return f"The result of {a} divided by {b} is {result}"


def demo_tool_with_errors():
    """Demo tool error handling."""

    tools = [divide]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        return "tools"

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")

    agent = graph.compile()

    print("\nTool Error Handling Demo:\n")

    queries = [
        "Divide 100 by 5",
        "Divide 100 by 0",  # Will trigger error
    ]

    for query in queries:
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        print(f"Query: {query}")
        print(f"Response: {result['messages'][-1].content}")
        print("-" * 40)


if __name__ == "__main__":
    # demo_tool_agent()
    # demo_tool_execution_trace()
    demo_tool_with_errors()
