from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("08_text_splitters/Project.md")

pages = loader.load()

# print(pages)

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(pages)

print("Chunks:", len(chunks))

for chunk in chunks:
    print("-" * 100)
    print(chunk.page_content)
