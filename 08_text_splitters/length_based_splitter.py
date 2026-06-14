from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# loader = PyPDFLoader("07_document_loaders/lasso.pdf")
loader = TextLoader("07_document_loaders/conclusion.txt")
pages = loader.load()



text_splitter = CharacterTextSplitter(separator=" ", chunk_size=300, chunk_overlap=30)

chunks = text_splitter.split_documents(pages)

print("Pages:", len(pages))
print("Chunks:", len(chunks))

for chunk in chunks:
    print("-"*100)
    print(chunk.metadata)
    print(chunk.page_content)
    print("-"*100)


