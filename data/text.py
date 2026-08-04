from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("data/RAG_Practice_Sample.pdf")
documents = loader.load()

# Create Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split Documents into Chunks
chunks = text_splitter.split_documents(documents)

# Print Information
print("=" * 50)
print("Total Pages:", len(documents))
print("Total Chunks:", len(chunks))
print("=" * 50)

# First Chunk
print("\nFirst Chunk:\n")
print(chunks[0].page_content)

# Metadata
print("\nMetadata:\n")
print(chunks[0].metadata)