import streamlit as st
import openai
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded properly
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")
else:
    print(f"API Key Loaded: {openai_api_key[:5]}...")  # Print part of the API key for verification

# Initialize OpenAI API key
openai.api_key = openai_api_key

# Streamlit page configuration
st.set_page_config(page_title="RAG-based Chatbot", layout="centered")

# Load documents or knowledge base for retrieval
def load_documents():
    documents = [
        {"text": "Streamlit is an open-source app framework for Machine Learning and Data Science teams."},
        {"text": "OpenAI offers powerful language models like GPT-4 for natural language understanding and generation."},
        {"text": "FAISS is a library for efficient similarity search and clustering of dense vectors."}
    ]
    return documents

# Create embeddings for the documents
def create_embedding_store(documents):
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Pass API key
    vectorstore = FAISS.from_texts([doc['text'] for doc in documents], embedding_model)
    return vectorstore

# Create the Retrieval Augmented Generation (RAG) chain
def create_rag_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    prompt = PromptTemplate(
        input_variables=["query"],
        template="You are an AI assistant. Answer the following question using the provided documents:\n\nQuery: {query}\n"
    )

    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return rag_chain

# Main function to handle the Streamlit UI
def chatbot_ui():
    st.title("RAG-based Chatbot using OpenAI")
    st.write("Ask any question based on the knowledge base.")

    documents = load_documents()
    vectorstore = create_embedding_store(documents)
    rag_chain = create_rag_chain(vectorstore)

    user_query = st.text_input("Enter your query", "")

    if user_query:
        response = rag_chain({"query": user_query})

        st.write("### Answer:")
        st.write(response["result"])

        st.write("### Source Documents:")
        for doc in response["source_documents"]:
            st.write(f"- {doc.page_content}")

# Run the UI function in Streamlit
if __name__ == "__main__":
    chatbot_ui()
