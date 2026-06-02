from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = PromptTemplate.from_template("What is the capital of {country}?")
 
country = input("Enter a country name: ")

prompt = template.invoke({"country": country})

result = model.invoke(prompt)

print(result.content)
