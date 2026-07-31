from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Paragraph 1

Python is easy.
Python is powerful.

Paragraph 2

Python is used in AI.
Python is used in Web Development.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=10
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}")
    print("-" * 30)
    print(chunk)