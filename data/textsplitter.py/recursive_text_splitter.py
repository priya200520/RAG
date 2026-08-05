from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load PDF
loader = PyPDFLoader("data/RAG_Practice_Sample.pdf")
documents = loader.load()

# Step 2: Create Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Step 3: Split the documents into chunks
chunks = text_splitter.split_documents(documents)

# Step 4: Print total chunks
print("=" * 50)
print("Total Chunks:", len(chunks))
print("=" * 50)

# Step 5: Print first chunk
print("\nFirst Chunk:\n")
print(chunks[0].page_content)

# Step 6: Print metadata
print("\nMetadata:\n")
print(chunks[0].metadata)

# Step 7: Print all chunks
print("\nAll Chunks:\n")

for i, chunk in enumerate(chunks):
    print("=" * 50)
    print(f"Chunk {i+1}")
    print("=" * 50)
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)