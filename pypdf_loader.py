from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("data/RAG_Practice_Sample.pdf")

documents = loader.load()
print(type(documents))