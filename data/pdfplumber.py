from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("data/python.pdf")
documents = loader.load()

print(type(documents))
print(len(documents))
print(documents[0].page_content[:300])
print(documents[0].metadata)
