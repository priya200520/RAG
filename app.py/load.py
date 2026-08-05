import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Load Environment Variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI Student Teacher",
    page_icon="🎓"
)

st.title("🎓 AI Student Teacher")
st.write("Ask questions from your uploaded PDF.")

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma Database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever()

# Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are an AI Student Teacher.

Answer only from the given context.

Context:
{context}

Question:
{input}
""")

# Create Chains
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# User Input
question = st.text_input("Enter your question")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": question})

        st.success("Answer")
        st.write(response["answer"])