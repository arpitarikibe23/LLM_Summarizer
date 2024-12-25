import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vectorize_documents import embeddings  # Import embeddings from the vectorization script

# Set up working directory and API configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
os.environ["GROQ_API_KEY"] = config_data["GROQ_API_KEY"]

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

def chat_chain(vectorstore):
    from langchain_groq import ChatGroq  # Import the LLM class

    llm = ChatGroq(
        model="llama-3.1-70b-versatile",  # Replace with your LLM of choice
        temperature=0  # Set low temperature to reduce hallucinations
    )
    retriever = vectorstore.as_retriever()  # Retrieve relevant chunks
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

    # Build the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # Define how documents are combined
        memory=memory,
        verbose=True,
        return_source_documents=True
    )
    return chain

# Streamlit UI
st.title("Bhagavad Gita & Yoga Sutras Query Assistant")

vectorstore = setup_vectorstore()
chain = chat_chain(vectorstore)

# User input
user_query = st.text_input("Ask a question about the Bhagavad Gita or Yoga Sutras:")
if user_query:
    response = chain.run(user_query)
    st.write(f"**Answer:** {response['answer']}")
    st.write(f"**Source Documents:** {response['source_documents']}")



