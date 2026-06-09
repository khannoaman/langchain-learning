from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough




load_dotenv()

def word_count(text):
    return len(text.split())


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = StrOutputParser()

prompt = PromptTemplate(template="Give a funny joke about {topic}.", input_variables=["topic"])

joke_gen_chain = prompt | model | parser

parallel = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
})

chain = joke_gen_chain | parallel


topic = input("Enter a topic for the joke: ")
result = chain.invoke({"topic": topic})

print("{} \n word count: {}".format(result["joke"], result["word_count"]))