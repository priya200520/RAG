from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

documents = [
    Document(page_content="Python is a programming language."),
    Document(page_content="LangChain helps build LLM applications."),
    Document(page_content="RAG improves AI responses by retrieving relevant information.")
]

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    collection_name="my_collection",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)

db.add_documents(documents)

print("✅ Documents Added Successfully!")