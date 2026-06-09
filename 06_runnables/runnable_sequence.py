from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = JsonOutputParser()

template = PromptTemplate(
    template="What are the top 3 movies of the following bollywood actor? {actor_name}\n\n{format_instructions}",  
    input_variables=["actor_name"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = RunnableSequence(template, model, parser)

actor_name = input("Enter the name of the Bollywood actor: ")

result = chain.invoke({"actor_name": actor_name})

print(result)