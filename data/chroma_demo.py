from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

db = Chroma(
    collection_name="my_collection",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)

print("ChromaDB Created Successfully!")