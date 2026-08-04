from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Existing ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

# Create Retriever
retriever = vectorstore.as_retriever()

# Search
query = "What is Artificial Intelligence?"

docs = retriever.invoke(query)

print("=" * 50)
print("Retrieved Documents")
print("=" * 50)

for doc in docs:
    print(doc.page_content)
    print("-" * 50)