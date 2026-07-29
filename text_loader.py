from langchain_community.document_loaders import TextLoader

loader = TextLoader("notes.txt")

documents = loader.load()
for document in documents:
   # print(document.page_content)
    #print(document.metadata)

#print(type(documents))
#print(len(documents))
#print(documents[0])
#print(documents[0].page_content)
print (documents[0].metadata)