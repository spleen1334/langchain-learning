# LangSmith Overview

## What it is

LangSmith is the observability and evaluation platform for LLM apps. It records every model call as a nested **run tree** — inputs, outputs, latency, token counts, cost, errors — and adds datasets + experiment comparison on top so you can tell whether a prompt or model change actually improved anything.

It's independent of LangChain in principle (`@traceable` works on any Python function), but LangChain/LangGraph auto-instrument themselves when tracing is enabled.

## Setup

Environment variables (in `.env`, never committed):

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=My Project        # optional, groups runs
```

That's it — any LCEL chain or compiled graph invoked afterwards is traced. Set `LANGSMITH_TRACING=false` to turn it off (`02_langgraph_control_flow/01_langgraph_core.py` has a note about toggling it per-demo).

→ [`05_production_patterns/01_langsmith_setup.py`](../05_production_patterns/01_langsmith_setup.py)

## Core concepts

### Tracing and run trees
A run is one unit of work. Chains nest their component runs, so a graph invocation produces a tree you can drill into: which node ran, what prompt it built, what the model returned, how long each step took.

### `@traceable`
Decorate arbitrary Python to make it a named run in the tree:

```python
@traceable(name="secure_process", tags=["production"])
def process(user_input: str) -> dict: ...
```
Used throughout section 5 to wrap non-LCEL classes (`ModelRouter.invoke`, `CachedLLM.invoke`, `SecurityGuard.check`, `InstrumentedLLM.invoke`) and in `projects/01_smart_bot_section1.py`. `run_type="chain"` and `tags=[...]` make traces filterable in the dashboard.

### Metadata and tags
Attach `user_id`, request type, environment, experiment version — then filter/segment in the UI. This is how you answer "which user hit the timeout" without shipping logs.

### Monitoring / metrics
LangSmith gives you latency, error rate, and token/cost dashboards per project. `05_production_patterns/02_monitoring.py` shows the complementary self-hosted half: structured JSON logs for a log aggregator and a `MetricsCollector` computing error rate, avg latency, token totals and cache hit rate — the numbers you'd export to Prometheus/Datadog and alert on.

→ [`05_production_patterns/02_monitoring.py`](../05_production_patterns/02_monitoring.py)

### Cost tracking
Traces carry token counts per call, so cost rolls up per run, per chain, per project. The application-side levers for reducing that number — complexity-based model routing, response caching, and per-request token budgets — are in `05_production_patterns/03_cost_optimization.py`, each wrapped in `@traceable` so the effect is visible in the dashboard.

→ [`05_production_patterns/03_cost_optimization.py`](../05_production_patterns/03_cost_optimization.py)

### Datasets and evaluation
The production testing loop:

1. **Dataset** — `client.create_dataset(...)` + `client.create_example(inputs={...}, outputs={...})`. A versioned, persistent set of test cases; you can also promote real production traces into it.
2. **Target** — a function `inputs: dict -> outputs: dict` wrapping the chain under test.
3. **Evaluators** — functions `(run, example) -> {"key": ..., "score": ...}`. Can be deterministic (keyword overlap) or LLM-as-judge (correctness against a reference, helpfulness without one).
4. **Run** — `evaluate(target, data=dataset_name, evaluators=[...], experiment_prefix="qa-chain-v1", max_concurrency=2)`.
5. **Compare** — re-run with a changed prompt/model under a different `experiment_prefix`, then diff the two experiments side by side in the dashboard.

→ [`05_production_patterns/05_testing_patterns.py`](../05_production_patterns/05_testing_patterns.py)

### The testing ladder
LLM output isn't `assert ==`-testable, so use layers:
- **Unit** — mock the LLM (`Mock().invoke.return_value = AIMessage(content="Paris")`); fast, free, deterministic; tests *your* glue code.
- **Integration** — real model, assert on substrings/properties rather than exact text.
- **Evaluation** — scored by an LLM judge against criteria (correctness, relevance, clarity, completeness) with a pass threshold.
- **Regression** — the evaluation run over a fixed dataset, compared against the previous experiment.

## Where demonstrated

| Concept | File |
|---|---|
| Enabling tracing, `@traceable`, named/tagged runs, metadata | `05_production_patterns/01_langsmith_setup.py` |
| Structured logging, metrics collection, instrumented LLM | `05_production_patterns/02_monitoring.py` |
| Model routing, caching, token budgeting (all traced) | `05_production_patterns/03_cost_optimization.py` |
| Mocks, integration tests, LLM-as-judge, datasets, `evaluate()`, experiment comparison | `05_production_patterns/05_testing_patterns.py` |
| Security pipeline steps as traced runs | `05_production_patterns/04_security_patterns.py` |
| Project-level tracing setup from env | `projects/01_smart_bot_section1.py` |
| Traced fallback chain | `02_langgraph_control_flow/07_error_handling.py` |
