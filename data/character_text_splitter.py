from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# Load PDF
loader = PyPDFLoader("data/python.pdf")
documents = loader.load()

# Create Text Splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100
)

# Split Documents
chunks = text_splitter.split_documents(documents)

# Print Output
print(f"Total Chunks: {len(chunks)}")
print("-" * 50)
print(chunks[0].page_content)
print("-" * 50)
print(chunks[0].metadata)