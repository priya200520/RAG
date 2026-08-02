from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Step 1: Create Documents
documents = [
    Document(page_content="Python is a programming language."),
    Document(page_content="LangChain helps build LLM applications."),
    Document(page_content="RAG improves AI responses by retrieving relevant information.")
]

# Step 2: Load Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Create ChromaDB
db = Chroma(
    collection_name="my_collection",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)

# Step 4: Add Documents to ChromaDB
db.add_documents(documents)

# Step 5: Convert ChromaDB into Retriever
retriever = db.as_retriever()

# Step 6: Ask a Query
query = "What is Python?"

# Step 7: Retrieve Similar Documents
results = retriever.invoke(query)

# Step 8: Print Results
print(f"\nQuery: {query}")

for i, doc in enumerate(results):
    print(f"\nResult {i+1}")
    print("-" * 40)
    print(doc.page_content)