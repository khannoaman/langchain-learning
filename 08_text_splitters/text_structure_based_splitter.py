from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# loader = PyPDFLoader("07_document_loaders/lasso.pdf")
loader = TextLoader("07_document_loaders/conclusion.txt")
pages = loader.load()


text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", ". ", " ", ""], 
                                                chunk_size=1000, chunk_overlap=500)

chunks = text_splitter.split_documents(pages)

print("Pages:", len(pages))
print("Chunks:", len(chunks))

for chunk in chunks:
    print("-"*100)
    print(chunk.metadata)
    print(chunk.page_content)
    print("-"*100)


