from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader

load_dotenv()  

loader = CSVLoader(file_path = "07_document_loaders/50_startups.csv")

docs = loader.load()

data = "\n\n".join([doc.page_content for doc in docs])

prompt = PromptTemplate(
    template = "Analyse the following data about 50 startups and present insights: {data}",
    input_variables = ["data"]
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"data": data})

print(result)