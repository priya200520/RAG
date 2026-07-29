from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="data/students.csv")

documents = loader.load()

print(type(documents))
print(len(documents))
print(documents[0].page_content)
print(documents[0].metadata)