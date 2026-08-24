"""
Error Handling and Reliability Patterns
Building robust LangGraph applications
"""

import operator
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from typing_extensions import TypedDict

load_dotenv()

# #######################
# === Retry Decorator ===
# #######################


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
):
    """Retry decorator with exponential backoff."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception: BaseException | None = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        # Exponential backoff, capped so a long retry chain can't sleep forever.
                        delay = min(base_delay * (2**attempt), max_delay)
                        # Jitter (0.5x–1.5x) de-synchronises many clients retrying at once,
                        # which otherwise re-creates the traffic spike that caused the failure.
                        delay = delay * (0.5 + random.random())
                        print(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)

            # No sleep after the final attempt; the original exception is re-raised so
            # the caller sees the real cause rather than a generic "retries exhausted".
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("with_retry: no attempts were made (max_retries <= 0)")

        return wrapper

    return decorator


@with_retry(max_retries=3, base_delay=1.0)
def unreliable_api_call(query: str) -> str:
    """Simulates an unreliable API."""
    if random.random() < 0.5:
        raise ConnectionError("Simulated API failure")
    return f"Success: {query}"


def demo_retry_pattern():
    """Demonstrate retry with exponential backoff."""

    print("Retry Pattern Demo:\n")

    for i in range(3):
        try:
            result = unreliable_api_call(f"Query {i}")
            print(f"✅ {result}")
        except Exception as e:
            print(f"❌ Failed after retries: {e}")


# #######################
# === Circuit Breaker ===
# #######################


class CircuitBreaker:
    """Circuit breaker pattern for failing services."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        # State machine:
        #   closed    = normal, calls pass through and failures are counted
        #   open      = tripped, calls are rejected instantly (no load on a dying service)
        #   half-open = probation after recovery_timeout; ONE trial call decides whether
        #               to close again (success) or re-open (failure)
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""

        # Check if circuit should move from open to half-open
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                # Fail fast without calling the service — that's the point of the breaker.
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            # Success - reset on half-open
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0

            return result

        # Catch-all on purpose: the breaker treats ANY failure of the wrapped call as a
        # health signal, then re-raises untouched so the caller still sees the real error.
        except Exception:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.state = "open"

            raise


def demo_circuit_breaker():
    """Demonstrate circuit breaker pattern."""

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)

    def flaky_service():
        if random.random() < 0.7:
            raise Exception("Service error")
        return "OK"

    print("\nCircuit Breaker Demo:\n")

    for i in range(15):
        try:
            result = breaker.call(flaky_service)
            print(f"Attempt {i + 1}: ✅ {result} (state: {breaker.state})")
        except Exception as e:
            print(f"Attempt {i + 1}: ❌ {e} (state: {breaker.state})")

        # After attempt 7, wait long enough for recovery
        if i == 6:
            print("  ⏳ Waiting 6 seconds for recovery timeout...")
            time.sleep(6)
        else:
            time.sleep(0.5)


# ############################
# === Model Fallback Chain ===
# ############################


class FallbackChain:
    """Try multiple models in order until one succeeds."""

    def __init__(self):
        self.models = [
            ("gpt-4o-mini", ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=10)),
            ("gpt-4o", ChatOpenAI(model="gpt-4o", temperature=0, timeout=10)),
            (
                "claude-sonnet",
                ChatAnthropic(
                    model_name="claude-sonnet-4-5-20250929",
                    temperature=0,
                    timeout=10,
                    stop=None,
                ),
            ),
        ]
        self.cache = {}

    # @traceable sends this function to LangSmith as a single named span, so the
    # fallbacks/cache hits show up as one logical operation in the trace tree.
    @traceable(name="fallback_invoke")
    def invoke(self, query: str, use_cache: bool = True) -> tuple[str, str]:
        """
        Invoke with fallbacks.
        Returns: (response, model_used)
        """

        # Check cache first
        if use_cache and query in self.cache:
            return self.cache[query], "cache"

        errors = []

        for model_name, model in self.models:
            try:
                response = model.invoke(query)
                result = response.content

                # Cache successful response
                self.cache[query] = result

                return result, model_name

            # Errors are collected rather than raised so the NEXT model gets a chance;
            # only if every provider fails do we surface the accumulated list.
            except Exception as e:
                errors.append(f"{model_name}: {e!s}")
                continue

        # All models failed
        raise Exception(f"All models failed: {errors}")


def demo_fallback_chain():
    """Demonstrate fallback chain."""

    chain = FallbackChain()

    print("\nFallback Chain Demo:\n")

    queries = [
        "What is 2 + 2?",
        "What is Python?",
        "What is 2 + 2?",  # Should hit cache
    ]

    for query in queries:
        try:
            result, model = chain.invoke(query)
            print(f"Query: {query}")
            print(f"  Model: {model}")
            print(f"  Response: {result[:50]}...")
        except Exception as e:
            print(f"Query: {query}")
            print(f"  ❌ Error: {e}")


# ################################
# === LangGraph Error Handling ===
# ################################


class RobustState(TypedDict):
    messages: Annotated[list, operator.add]
    error: str | None
    retry_count: int
    max_retries: int
    success: bool


def create_robust_agent():
    """Create agent with built-in error handling."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def process_with_retry(state: RobustState) -> dict:
        """Process with retry logic built-in."""

        try:
            # Simulate occasional failure
            if random.random() < 0.3 and state["retry_count"] < 2:
                raise Exception("Simulated processing error")

            response = llm.invoke(state["messages"])

            return {"messages": [response], "success": True, "error": None}

        # The exception is converted into STATE, not propagated. An exception escaping a
        # node aborts the whole graph; recording it lets the conditional edge below decide
        # whether to retry or degrade gracefully.
        except Exception as e:
            return {
                "error": str(e),
                "retry_count": state["retry_count"] + 1,
                "success": False,
            }

    def should_continue(state: RobustState) -> Literal["retry", "error", "success"]:
        if state["success"]:
            return "success"
        elif state["retry_count"] < state["max_retries"]:
            return "retry"
        else:
            return "error"

    # Graceful degradation: retries exhausted, so return a user-facing apology message
    # instead of letting the graph blow up.
    def handle_error(state: RobustState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content=f"I apologize, but I encountered an error: {state['error']}. "
                    "Please try again later."
                )
            ]
        }

    def finalize(state: RobustState) -> dict:
        return dict(state)

    # Build graph
    graph = StateGraph(RobustState)

    graph.add_node("process", process_with_retry)
    graph.add_node("handle_error", handle_error)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "process")
    graph.add_conditional_edges(
        "process",
        should_continue,
        # "retry" points back at the source node — retrying is just a self-loop edge.
        {"retry": "process", "error": "handle_error", "success": "finalize"},
    )
    graph.add_edge("handle_error", END)
    graph.add_edge("finalize", END)

    return graph.compile()


def demo_robust_agent():
    """Demonstrate robust agent with error handling."""

    agent = create_robust_agent()

    print("\nRobust Agent Demo:\n")

    for i in range(3):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Hello!")],
                "error": None,
                "retry_count": 0,
                "max_retries": 3,
                "success": False,
            }
        )

        status = "✅ Success" if result["success"] else "❌ Failed"
        print(f"Attempt {i + 1}: {status}")
        print(f"  Retries used: {result['retry_count']}")
        print(f"  Response: {result['messages'][-1].content[:50]}...")


if __name__ == "__main__":
    # Example usage of the unreliable API call with retry logic
    # try:
    #     result = unreliable_api_call("Hello, World!")
    #     print(result)
    # except Exception as e:
    #     print(f"API call failed after retries: {e}")

    # Run the retry pattern demonstration
    # demo_retry_pattern()

    # Run the circuit breaker demonstration
    demo_circuit_breaker()

    # Run the fallback chain demonstration
    # demo_fallback_chain()

    # Run the robust agent demonstration
    # demo_robust_agent()
