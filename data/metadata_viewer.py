from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/RAG_Practice_Sample.pdf")
documents = loader.load()

for doc in documents:
    print(doc.metadata)