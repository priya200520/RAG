from langchain_community.document_loaders import PyPDFLoader

# Load the PDF
loader = PyPDFLoader("data/RAG_Practice_Sample.pdf")
documents = loader.load()

# Print document information
print("=" * 50)
print("📄 PDF Information")
print("=" * 50)

print(f"Total Pages: {len(documents)}")

for i, doc in enumerate(documents):
    print(f"\nPage Number: {i + 1}")
    print(f"Metadata: {doc.metadata}")
    print(f"Characters: {len(doc.page_content)}")
    print("-" * 50)