from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/RAG_Practice_Sample.pdf")
documents = loader.load()

print(f"Total Pages: {len(documents)}")

for i in range(len(documents)):
    print(f"Page {i+1} Loaded Successfully")