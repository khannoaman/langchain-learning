from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2", task="feature-extraction")


text = "What is the capital of India?"

result = embeddings.embed_query(text)

print(result)

