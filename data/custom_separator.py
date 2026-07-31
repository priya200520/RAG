from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Python is easy to learn.

It is used in AI.

It is also used in Web Development.

It supports Object-Oriented Programming.

Python is one of the most popular programming languages.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n{'='*50}")
    print(f"Chunk {i+1}")
    print('='*50)
    print(chunk)