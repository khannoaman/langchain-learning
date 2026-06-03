from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

template1 = PromptTemplate.from_template("Create a funny joke about {topic}.")

template2 = PromptTemplate.from_template("Explain the joke in simple terms : {joke}")

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

topic = input("Enter a topic for the joke: ")

result = chain.invoke({"topic": topic})

print(result)