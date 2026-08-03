from langchain_community.document_loaders import PyPDFLoader

# Step 1: Create PDF Loader
loader = PyPDFLoader("data/RAG_Practice_Sample.pdf")

# Step 2: Load the PDF
documents = loader.load()

# Step 3: Check the data type
print("Type:", type(documents))

# Step 4: Count total pages
print("Total Pages:", len(documents))

# Step 5: Print first document
print("\nFirst Document:\n")
print(documents[0])

# Step 6: Print first page content
print("\nPage Content:\n")
print(documents[0].page_content)

# Step 7: Print metadata
print("\nMetadata:\n")
print(documents[0].metadata)