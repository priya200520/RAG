from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = PyPDFLoader("data/RAG_Practice_Sample.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(chunks[0].page_content)
print(chunks[0].metadata)