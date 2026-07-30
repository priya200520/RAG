from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# Load PDF
loader = PyPDFLoader("data/python.pdf")
documents = loader.load()

print("=" * 60)
print("Original Documents:", len(documents))
print("=" * 60)

# Create Splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,
    chunk_overlap=50
)

# Split
chunks = text_splitter.split_documents(documents)

print("Total Chunks:", len(chunks))
print("=" * 60)

# Print first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n📄 Chunk {i+1}")
    print("-" * 50)
    print(chunk.page_content)
    print("-" * 50)
    print(chunk.metadata)