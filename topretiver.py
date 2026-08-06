from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma Database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Create Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# User Query
query = input("Enter your question: ")

# Retrieve Top 3 Chunks
documents = retriever.invoke(query)

print("=" * 60)
print("Top 3 Retrieved Chunks")
print("=" * 60)

for i, doc in enumerate(documents, start=1):
    print(f"\nChunk {i}")
    print("-" * 40)
    print(doc.page_content)
    print("\nMetadata:")
    print(doc.metadata)