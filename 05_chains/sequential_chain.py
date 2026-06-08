from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel, Field

class Summary(BaseModel):
    key_points1: str = Field(description="The first key point summarizing the report")
    key_points2: str = Field(description="The second key point summarizing the report")
    key_points3: str = Field(description="The third key point summarizing the report")
    key_points4: str = Field(description="The fourth key point summarizing the report")
    key_points5: str = Field(description="The fifth key point summarizing the report")

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

str_parser = StrOutputParser()
json_parser = PydanticOutputParser(pydantic_object=Summary)

template1 = PromptTemplate(
    template = "Write a detailed report on the {topic}.",
    input_variables=["topic"]
)   

template2 = PromptTemplate(
    template = "Summarize the following report in 5 key points:\n\n{report} \n\n{format_instructions}",
    input_variables=["report"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()}
)


chain = template1 | model | str_parser | template2 | model | json_parser

topic = input("Enter a topic to generate a report: ")

result = chain.invoke({"topic": topic})

print(result)