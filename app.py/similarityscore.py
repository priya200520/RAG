from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

query = "Explain Machine Learning"

results = vectorstore.similarity_search_with_score(query)

for doc, score in results:

    print("=" * 50)
    print("Similarity Score:", score)
    print(doc.page_content)