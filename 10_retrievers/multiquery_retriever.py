from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import WikipediaRetriever
import wikipedia
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

wikipedia.set_user_agent("LangChainPracticeBot/1.0 (noaman@example.com)")
wiki_retriever = WikipediaRetriever(top_k_results=3,lang="en", doc_content_chars_max=800)

embeddings = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2", task="feature-extraction")


teams = ["Royal Challenger Banglore","Chennai Super Kings","Mumbai Indians","Sunrisers Hyderabad","Delhi Capitals","Kolkata Knight Riders","Punjab Kings"]
docs = []
for team in teams:
    query = f"IPL '{team}'"
    docs += wiki_retriever.invoke(query)

print(f"Total {len(docs)} documents.\n")


vectorstore = FAISS.from_documents(docs, embeddings)

# Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 5,"lambda_mult": 0.8})
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model)


query = "Provide all the information about IPL 2026. Who won the league and information about different awards?"


results = retriever.invoke(query)

print(f"Found {len(results)} documents from vector store.\n")

for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(f"Title: {doc.metadata.get('title', 'Unknown')}")
    print(f"Content snippet: {doc.page_content.strip()}...\n")
    print(f"Source URL: {doc.metadata.get('source', 'Unknown')}")
    print("-" * 50 + "\n")

results_multi = multi_query_retriever.invoke(query)

print(f"Found {len(results_multi)} documents from vector store.\n")

for i, doc in enumerate(results_multi, 1):
    print(f"--- Result {i} ---")
    print(f"Title: {doc.metadata.get('title', 'Unknown')}")
    print(f"Content snippet: {doc.page_content.strip()}...\n")  
    print(f"Source URL: {doc.metadata.get('source', 'Unknown')}")
    print("-" * 50 + "\n")