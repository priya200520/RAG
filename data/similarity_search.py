from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Create Documents
documents = [
    Document(page_content="Python is a programming language."),
    Document(page_content="LangChain helps build LLM applications."),
    Document(page_content="RAG improves AI responses by retrieving relevant information.")
]

# Load Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create ChromaDB
db = Chroma(
    collection_name="my_collection",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)

# Add Documents
db.add_documents(documents)

# Search
results = db.similarity_search("What is Python?")

# Print Results
for i, doc in enumerate(results):
    print(f"\nResult {i+1}")
    print("-" * 30)
    print(doc.page_content)