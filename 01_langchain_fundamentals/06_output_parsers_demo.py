from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model(model="gpt-4o-mini", temperature=0)


def demo_str_output_parser():
    """StrOutputParser: unwraps the AIMessage into a plain string."""
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template("wire a short poem about {topic}")
    chain = prompt | llm | parser

    response = chain.invoke({"topic": "nature"})
    # <class 'str'> — without StrOutputParser this would be an AIMessage.
    print(type(response))


def demo_json_output_parser():
    """JsonOutputParser: parses the text into a plain dict. Strips ```json
    fences, but does NOT validate the shape — missing/extra keys pass through
    silently."""
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name' and 'age' for: {description}"
    )
    chain = prompt | llm | parser

    result = chain.invoke({"description": "A 25-year-old developer named Alex"})
    print(result)  # {'name': 'Alex', 'age': 25}


class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's occupation")


def demo_pydantic_output_parser():
    """PydanticOutputParser: get_format_instructions() renders the Pydantic
    JSON schema as prompt text, and the parser validates the model's reply
    against that schema."""
    parser = PydanticOutputParser(pydantic_object=Person)
    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name', 'age', and 'occupation' for: {description}\n"
        "{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser

    result = chain.invoke({"description": "A 30-year-old artist named Maria"})
    print(result)  # Person(name='Maria', age=30, occupation='artist')


class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The rating of the movie out of 10")


def demo_with_structured_output():
    """with_structured_output: preferred over PydanticOutputParser — the
    schema is sent to the provider as a tool/JSON-schema constraint, so the
    model is forced to emit valid fields rather than being asked nicely in
    the prompt and parsed afterwards. No format instructions needed."""
    structured_model = llm.with_structured_output(MovieReview)

    result = structured_model.invoke(
        "Review: Inception is a mind-bending thriller. 9/10"
    )
    print(
        result
    )  # MovieReview(title='Inception', review='A mind-bending thriller.', rating=9)


if __name__ == "__main__":
    demo_str_output_parser()
    demo_json_output_parser()
    demo_pydantic_output_parser()
    demo_with_structured_output()
