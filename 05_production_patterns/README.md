# 05 — Production Patterns

> **Numeric prefixes = suggested reading order** (increasing difficulty; files are independent). One soft dependency: `01_langsmith_setup.py` turns on the tracing that `02`–`05` all decorate with `@traceable`.

What has to exist around an agent before it faces real users: tracing, metrics, cost control, security, and automated evaluation.

## `01_langsmith_setup.py`

**What it does**
- Enables tracing with `LANGSMITH_TRACING=true`.
- Wraps functions in `@traceable(name=..., tags=[...])` so arbitrary Python — not just LCEL chains — shows up as a named run tree in LangSmith.
- Demonstrates naming runs and attaching metadata for later filtering.

**What it's for**
- Observability baseline — once tracing is on, every chain invocation, token count, latency and error is recorded without further code changes.

## `02_monitoring.py`

**What it does** — self-rolled telemetry:
- A `JSONFormatter` logging subclass emitting structured JSON logs (timestamp/level/module/function + `extra_data`) for log aggregators.
- A `MetricsCollector` tracking requests, errors, latency, token counts and cache hits, with derived error rate / avg latency / hit rate.
- An `InstrumentedLLM` wrapper combining timing, metrics recording, structured logging and `@traceable`.

**What it's for**
- The metrics you actually get paged on — p99 latency, error rate, token burn — in a form Datadog/CloudWatch can index.

## `03_cost_optimization.py`

**What it does** — three levers:
- `ModelRouter` classifies query complexity with a cheap model, then dispatches simple queries to gpt-4o-mini and complex ones to gpt-4o, returning an estimated cost.
- `SemanticCache` / `CachedLLM` hash-normalize queries for exact-match caching and report hit rate (with notes on the embedding-similarity version).
- `TokenBudget` / `BudgetedLLM` estimate tokens per request, reject over-budget calls, and accumulate usage stats.

**What it's for**
- Keeping the bill sane — most production traffic is repetitive and simple, so routing plus caching is usually a large cost cut before any prompt tuning.

## `04_security_patterns.py`

**What it does** — layered defense:
- `InputSanitizer` regex-matches prompt-injection phrasings ("ignore all previous instructions", "---END OF PROMPT---") and strips delimiter runs.
- `PIIDetector` finds/masks email, phone, SSN, credit card and IP with regex.
- `SecurityGuard` is the LLM-as-judge classifier returning `{"safe": bool, "reason": ...}`, failing closed on parse error.
- `OutputValidator` re-scans model output for leaked PII and harmful patterns.
- `SecurePipeline` chains all five steps around the actual LLM call.

**What it's for**
- Prompt injection, data exfiltration, and PII compliance.
- The checks that must run on both sides of the model, not just on input.

## `05_testing_patterns.py`

**What it does** — the testing ladder:
- Unit tests with a `Mock` LLM returning a canned `AIMessage` (fast, deterministic, no API cost).
- Integration tests asserting real responses contain expected substrings.
- `LLMEvaluator` scoring correctness/relevance/clarity/completeness as LLM-as-judge.
- A `RegressionTestRunner` over a test-case list with a pass threshold.

Second half is the production approach:
- LangSmith `Client` datasets (`create_dataset`/`create_example`).
- A `@traceable` target function.
- Custom evaluators (`correctness`, `helpfulness`, keyword-overlap `contains_answer`).
- `evaluate(..., experiment_prefix=...)` run twice with different prompts, so v1 vs v2 can be compared in the dashboard.

**What it's for**
- Knowing whether a prompt/model change made things better.
- LLM outputs aren't assert-equal testable, so you need scored datasets and experiment comparison instead of unit tests alone.
