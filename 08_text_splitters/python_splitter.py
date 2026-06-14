from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("08_text_splitters/test_python_splitter.py")

pages = loader.load()

print(len(pages))

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

print("Chunks:", len(chunks))

for chunk in chunks:
    print("-" * 100)
    print(chunk.page_content)
