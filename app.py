import streamlit as st
import openai
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI for chat models
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv  # Import dotenv to load .env files

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# If the API key is not set, raise an error
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

# Initialize your OpenAI API key
openai.api_key = openai_api_key

# Streamlit page configuration
st.set_page_config(page_title="RAG-based Chatbot", layout="centered")

# Load documents or knowledge base for retrieval
def load_documents():
    documents = [
        {"text": "Streamlit is an open-source app framework for Machine Learning and Data Science teams."},
        {"text": "OpenAI offers powerful language models like GPT-4 for natural language understanding and generation."},
        {"text": "XXX is a library for efficient similarity search and clustering of dense vectors."}
    ]
    return documents

# Create embeddings for the documents
def create_embedding_store(documents):
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Pass API key
    vectorstore = FAISS.from_texts([doc['text'] for doc in documents], embedding_model)
    return vectorstore

# Create the Retrieval Augmented Generation (RAG) chain
def create_rag_chain(vectorstore):
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)  # Use ChatOpenAI for chat models

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["query"],
        template="You are an AI assistant. Answer the following question using the provided documents:\n\nQuery: {query}\n"
    )

    # Set up the retriever
    retriever = vectorstore.as_retriever()

    # Build the RetrievalQA chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" type combines all retrieved documents
        retriever=retriever,
        return_source_documents=True
    )

    return rag_chain

# Main function to handle the Streamlit UI
def chatbot_ui():
    st.title("RAG-based Chatbot using OpenAI")
    st.write("Ask any question based on the knowledge base.")

    # Load the documents
    documents = load_documents()

    # Create embeddings and vector store
    vectorstore = create_embedding_store(documents)

    # Create the RAG chain
    rag_chain = create_rag_chain(vectorstore)

    # User query input
    user_query = st.text_input("Enter your query", "")

    if user_query:
        # Run the RAG chain with the user's query
        response = rag_chain({"query": user_query})

        # Display the response
        st.write("### Answer:")
        st.write(response["result"])  # Extract the answer from the response

        # Display the source documents used for the answer
        st.write("### Source Documents:")
        for doc in response["source_documents"]:
            st.write(f"- {doc.page_content}")

# Run the UI function in Streamlit
if __name__ == "__main__":
    chatbot_ui()