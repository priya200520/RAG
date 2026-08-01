from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Python is easy to learn."

vector = embedding.embed_query(text)

print(f"Vector Length: {len(vector)}")
print(vector[:10])