from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

loader = TextLoader("07_document_loaders/conclusion.txt")
pages = loader.load()

embeddings = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2", task="feature-extraction")


semantic_splitter = SemanticChunker(
embeddings = embeddings,
 breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=98  
)

chunks = semantic_splitter.split_documents(pages)

print("Chunks:", len(chunks))

for chunk in chunks:
    print("-" * 100)
    print(chunk.page_content)