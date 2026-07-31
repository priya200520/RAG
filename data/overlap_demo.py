from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Python is a high-level programming language.
It is easy to learn.
It supports Object-Oriented Programming.
Python is widely used in Artificial Intelligence.
Python is also used in Web Development.
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20
)

chunks = text_splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n{'='*50}")
    print(f"Chunk {i+1}")
    print('='*50)
    print(chunk)