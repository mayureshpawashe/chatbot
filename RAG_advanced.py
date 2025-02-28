import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load API keys from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Function to fetch live news data
def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&category=technology&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        news_text = "\n".join([article["title"] + ": " + article["description"] for article in articles if article["description"]])
        return news_text
    return "Failed to fetch live news."

# Fetch news
news_text = fetch_news()

# Split text into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(news_text)

# Convert text into embeddings using Hugging Face
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_texts(chunks, embedding_model, persist_directory="./chroma_db")


# Initialize Groq LLM
llm = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=GROQ_API_KEY)

# Create Retrieval-based QA system
retriever = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Streamlit UI
st.title("Live News Q&A")
st.write("Ask questions based on the latest technology news.")

# User input for questions
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query:
        response = qa_chain.run(query)
        st.write("### Answer:")
        st.write(response)
    else:
        st.warning("Please enter a question.")
