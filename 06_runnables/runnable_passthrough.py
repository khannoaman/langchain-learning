from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

prompt1 = PromptTemplate(template="Give a funny joke about {topic}.", input_variables=["topic"])
prompt2 = PromptTemplate(template="Explain the given joke: {joke}.", input_variables=["joke"])

parallel = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": prompt2 | model | parser
})


chain = prompt1 | model | parser | parallel


topic = input("Enter a topic for the joke: ")
result = chain.invoke({"topic": topic})

print(result)
