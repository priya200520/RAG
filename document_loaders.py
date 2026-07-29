from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("data/RAG_Practice_Sample.pdf")
#pypdf_loader.py
documents = loader.load()
documents = loader.lazy_load()

#print(len(documents))
#print(documents[0])
# TextLoaader
print(documents[0].page_content)
