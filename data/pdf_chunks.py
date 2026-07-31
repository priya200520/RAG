from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("data/python.pdf")
documents = loader.load()

# Create Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Split PDF into chunks
chunks = splitter.split_documents(documents)

print(f"\nTotal Chunks: {len(chunks)}")

# Print all chunks
for i, chunk in enumerate(chunks):
    print("\n" + "=" * 60)
    print(f"Chunk {i+1}")
    print("=" * 60)
    print(chunk.page_content)