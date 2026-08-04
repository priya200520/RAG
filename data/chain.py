from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()

# Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever()

# Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are an AI Teacher.

Answer the question only from the provided context.

Context:
{context}

Question:
{input}
""")

# Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Ask Question
query = input("Ask Your Question: ")

response = qa_chain.invoke({"query": query})

print("\nAnswer:\n")
print(response["result"])