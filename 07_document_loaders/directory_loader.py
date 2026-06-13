from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

load_dotenv()  


loader = DirectoryLoader(path = "07_document_loaders/documents/",
                         glob = "*.pdf",
                         loader_cls = PyPDFLoader)

docs = loader.load()

for doc in docs:
    print(doc.metadata)
