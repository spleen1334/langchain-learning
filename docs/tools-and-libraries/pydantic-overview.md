# Pydantic Overview

## What it is

Pydantic is Python's standard library for **data validation and settings management** using type hints. You define a schema as a class with annotated fields; Pydantic validates, coerces, and parses input against it at runtime — and raises a structured `ValidationError` when it doesn't fit.

It's everywhere in this course because LangChain/LangGraph use it for structured LLM output, tool-call schemas, and graph state.

## Core concepts

### Defining a model
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str | None = None  # optional, defaults to None
```
- Fields with no default are required.
- Types are enforced and coerced where reasonable (`"30"` → `30` for an `int` field).

### Validating data
```python
user = User(name="Ana", age=30)          # OK
User(name="Ana", age="not a number")     # raises ValidationError
```
- `model_validate(dict)` — validate from a dict.
- `model_validate_json(str)` — validate from a raw JSON string.
- `model_dump()` / `model_dump_json()` — serialize back out.

### Field constraints
```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(min_length=1)
    price: float = Field(gt=0)
    tags: list[str] = Field(default_factory=list)
```
`Field(...)` adds validation rules (`gt`, `ge`, `lt`, `le`, `min_length`, `max_length`, `pattern`) and metadata (`description`) beyond the bare type.

### Custom validators
```python
from pydantic import field_validator

class Signup(BaseModel):
    password: str

    @field_validator("password")
    @classmethod
    def check_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password too short")
        return v
```
For cross-field checks, use `model_validator(mode="after")` instead.

### Nested models
Models can contain other models — Pydantic validates recursively:
```python
class Address(BaseModel):
    city: str
    zip_code: str

class User(BaseModel):
    name: str
    address: Address
```

### Why LangChain/LangGraph use it
- **Structured LLM output** — `model.with_structured_output(MySchema)` gets the model to return data matching a Pydantic model instead of free text.
- **Tool schemas** — a tool's Pydantic model becomes its input schema, so the model knows what arguments to produce and LangChain validates them before the tool runs.
- **Graph state** — LangGraph state can be defined as a Pydantic model so every node reads/writes a validated shape.

## Most common cases in practice
| Use case | How |
|---|---|
| Parse/validate an API request or LLM output | Define a `BaseModel`, call `model_validate` / `model_validate_json` |
| Force an LLM to return structured data | Pass the model to `with_structured_output()` |
| Define a tool's arguments | Use the model as the tool's `args_schema` |
| Reject bad input early | Raise inside a `field_validator` / `model_validator` |
| Convert model ↔ JSON | `model_dump_json()` / `model_validate_json()` |
