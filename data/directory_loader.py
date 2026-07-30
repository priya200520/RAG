from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

loader = DirectoryLoader(
    "data",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

print(type(documents))
print(len(documents))
print(documents[0].metadata)
print(documents[0].page_content[:200])