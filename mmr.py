from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Create Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma Vector Database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Create MMR Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)

# User Query
query = input("Enter your question: ")

# Retrieve Documents
docs = retriever.invoke(query)

print("=" * 60)
print("MMR Retrieved Documents")
print("=" * 60)

for i, doc in enumerate(docs, start=1):
    print(f"\nDocument {i}")
    print("-" * 40)
    print(doc.page_content)
    print("\nMetadata:")
    print(doc.metadata)