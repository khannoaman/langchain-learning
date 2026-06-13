from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

load_dotenv()  

def return_page_content(path):
    loader = TextLoader(path["path"])
    return loader.load()[0].page_content

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

path = "07_document_loaders/conclusion.txt"

prompt1 = PromptTemplate(
    template = "Summarise the following document: {document}",
    input_variables = ["document"]
)

prompt2 = PromptTemplate(
    template = "Generate 5 MCQs based on the following document: {document}",
    input_variables = ["document"]
)

parser = StrOutputParser()

parallel = RunnableParallel({
    "summary": prompt1 | model | parser,
    "mcqs": prompt2 | model | parser
}
)

chain = RunnableLambda(return_page_content) | parallel

result = chain.invoke({"path": path})

print(f"Summary\n\n {result['summary']} \n]n MCQS \n\n {result['mcqs']}")




