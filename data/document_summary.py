from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/RAG_Practice_Sample.pdf")
documents = loader.load()

print("===== DOCUMENT SUMMARY =====")
print("Total Pages:", len(documents))

for i, doc in enumerate(documents):
    print(f"\nPage {i+1}")
    print(doc.page_content[:200])   # First 200 characters