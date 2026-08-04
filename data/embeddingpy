from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("data/RAG_Practice_Sample.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

# Create Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert first chunk into embedding
vector = embedding.embed_query(chunks[0].page_content)

print("=" * 50)
print("Embedding Length:", len(vector))
print("=" * 50)
print(vector[:10])   # Print first 10 values