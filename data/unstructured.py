from langchain_community.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader("data/notes.txt")

documents = loader.load()

print(type(documents))
print(len(documents))
print(documents[0].page_content[:200])
print(documents[0].metadata)