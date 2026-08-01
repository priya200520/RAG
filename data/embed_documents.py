from langchain_huggingface import HuggingFaceEmbeddings

# Load Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Multiple Documents
documents = [
    "Python is a programming language.",
    "LangChain is used to build LLM applications.",
    "RAG improves the accuracy of AI responses."
]

# Convert documents into vectors
vectors = embedding.embed_documents(documents)

print("Total Documents:", len(documents))
print("Total Vectors:", len(vectors))
print("Length of First Vector:", len(vectors[0]))
print("\nFirst 10 values of First Vector:")
print(vectors[0][:10])