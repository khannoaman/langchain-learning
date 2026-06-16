from langchain_community.retrievers import WikipediaRetriever
import wikipedia

# Wikipedia's API blocks requests without a proper User-Agent header.
# We need to set a custom User-Agent to avoid receiving a 403 HTML error.
wikipedia.set_user_agent("LangChainPracticeBot/1.0 (noaman@example.com)")

# Initialize the Wikipedia retriever
# top_k_results: Number of Wikipedia pages to return (default is 3)
# doc_content_chars_max: Maximum number of characters to return per page (default is 4000)
# retriever = WikipediaRetriever(
#     top_k_results=3,
#     doc_content_chars_max=500 
# )

retriever = WikipediaRetriever(top_k_results=5,lang="en")

# Perform a search query
query = "Virat Kohli's best innings"
print(f"Searching Wikipedia for: '{query}'...\n")
docs = retriever.invoke(query)

print(f"Found {len(docs)} documents.\n")

for i, doc in enumerate(docs, 1):
    print(f"--- Result {i} ---")
    print(f"Title: {doc.metadata.get('title', 'Unknown')}")
    print(f"Content snippet: {doc.page_content.strip()}...\n")
    print(f"Source URL: {doc.metadata.get('source', 'Unknown')}")
    print("-" * 50 + "\n")