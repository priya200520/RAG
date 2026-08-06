from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Create Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma Database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Similarity Search
query = "What is Artificial Intelligence?"

results = vectorstore.similarity_search(
    query=query,
    k=2
)

print("=" * 60)
print("Top 2 Similar Documents")
print("=" * 60)

for i, doc in enumerate(results, start=1):
    print(f"\nDocument {i}")
    print("-" * 40)
    print(doc.page_content)
    print("\nMetadata:")
    print(doc.metadata)