from langchain_community.document_loaders import PyPDFLoader
import os

folder_path = "data"

documents = []

for file in os.listdir(folder_path):

    if file.endswith(".pdf"):

        loader = PyPDFLoader(os.path.join(folder_path, file))

        documents.extend(loader.load())

print("=" * 50)
print("Total Documents Loaded:", len(documents))
print("=" * 50)

for doc in documents:
    print(doc.metadata)