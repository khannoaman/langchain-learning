from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give details about the following cricket player: {player}\n\n{format_instructions}",
    input_variables=["player"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

player = input("Enter the name of the cricket player: ")

chain = template | model | parser

result = chain.invoke({"player": player})

print(result)
