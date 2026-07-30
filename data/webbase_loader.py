from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://python.org")

documents = loader.load()

print(type(documents))
print(len(documents))
print(documents[0].metadata)
print(documents[0].page_content[:300])
