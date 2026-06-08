from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = JsonOutputParser()

template = PromptTemplate(
    template = "provide top 5 destinations to visit in {country} and explain why they are worth visiting.\n\n{format_instructions}",
    input_variables=["country"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

country = input("Enter a country to get travel recommendations: ")

result = chain.invoke({"country": country})

print(result)

