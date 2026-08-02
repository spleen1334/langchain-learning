from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
)

load_dotenv()

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("wire a short poem about {topic}")

llm = init_chat_model(model="gpt-4o-mini", temperature=0)

chain = prompt | llm | parser

response = chain.invoke({"topic": "nature"})

# <class 'str'> — without StrOutputParser this would be an AIMessage.
print(type(response))


# JsonOutputParser example
from langchain_core.output_parsers import JsonOutputParser

# Parses the text into a plain dict. It strips ```json fences, but it does NOT
# validate the shape — missing/extra keys pass through silently.
parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with 'name' and 'age' for: {description}"
)

chain = prompt | llm | parser

result = chain.invoke({"description": "A 25-year-old developer named Alex"})
print(result)  # {'name': 'Alex', 'age': 25}

# PydanticOutputParser example
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's occupation")


parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with 'name', 'age', and 'occupation' for: {description}"
    # get_format_instructions() renders the Pydantic JSON schema as prompt text.
    # NOTE: this template has no {format_instructions} placeholder, so the partial is
    # inert here — it only takes effect if the placeholder is present in the template.
).partial(format_instructions=parser.get_format_instructions())
chain = prompt | llm | parser
result = chain.invoke({"description": "A 30-year-old artist named Maria"})
print(result)  # Person(name='Maria', age=30, occupation='artist')


# Structured Output
class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The rating of the movie out of 10")


# Preferred over PydanticOutputParser: the schema is sent to the provider as a
# tool/JSON-schema constraint, so the model is forced to emit valid fields rather
# than being asked nicely in the prompt and parsed afterwards. No format instructions needed.
structured_model = llm.with_structured_output(MovieReview)

result = structured_model.invoke("Review: Inception is a mind-bending thriller. 9/10")
print(result)  # MovieReview(title='Inception', review='A mind-bending thriller.', rating=9)