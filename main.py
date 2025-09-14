import os
import pickle
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


# -------------------- Environment Setup --------------------

# Load environment variables from .env file (e.g. OPENAI_API_KEY)
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Ensure API key is set for LangChain/OpenAI usage
os.environ["OPENAI_API_KEY"] = openai_api_key

# File path where FAISS vector index will be saved/loaded
INDEX_FILE = "faiss_index.pkl"


# -------------------- Helper Functions --------------------

def fetch_text_from_url(url):
    """
    Fetches raw text content from a webpage URL by extracting text from <p> tags.
    Returns a LangChain Document object with the extracted text and source URL metadata.

    Parameters:
        url (str): The URL to fetch and extract text from.

    Returns:
        Document or None: LangChain Document if successful, else None.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        
        # Join all paragraph texts with double newline separators
        text = "\n\n".join(p.get_text() for p in paragraphs if p.get_text().strip())
        
        if not text.strip():
            st.warning(f"No textual content found at URL: {url}")
        
        return Document(page_content=text, metadata={"source": url})
    except Exception as e:
        st.warning(f"Failed to fetch or parse URL '{url}': {e}")
        return None


def load_data_from_urls(url_list):
    """
    Given a list of URLs, fetches and returns a list of LangChain Document objects.

    Parameters:
        url_list (list): List of string URLs.

    Returns:
        list: List of Document objects with page content and metadata.
    """
    documents = []
    for url in url_list:
        doc = fetch_text_from_url(url)
        if doc:
            documents.append(doc)
    st.write(f"Loaded {len(documents)} documents from URLs.")
    return documents


def split_documents(documents):
    """
    Splits large documents into smaller chunks for better embedding and retrieval.

    Parameters:
        documents (list): List of LangChain Document objects.

    Returns:
        list: List of split Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],  # Preferred split separators in order
        chunk_size=1000,                      # Maximum chunk size in characters
        chunk_overlap=100                     # Overlap between chunks to maintain context
    )
    split_docs = splitter.split_documents(documents)
    st.write(f"Split documents into {len(split_docs)} chunks.")
    return split_docs


def create_faiss_index(documents, file_path=INDEX_FILE):
    """
    Creates a FAISS vector store from the given documents, saves it to disk, and returns the vector store.

    Parameters:
        documents (list): List of LangChain Document chunks.
        file_path (str): Path to save the FAISS index pickle file.

    Returns:
        FAISS vector store or None if failed.
    """
    if not documents:
        st.error("No documents available to create an index.")
        return None

    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)
        st.write(f"✅ FAISS index saved at: {os.path.abspath(file_path)}")
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create/save FAISS index: {e}")
        return None


def load_faiss_index(file_path=INDEX_FILE):
    """
    Loads a FAISS vector store index from disk.

    Parameters:
        file_path (str): Path to the FAISS index pickle file.

    Returns:
        FAISS vector store or None if failed.
    """
    if not os.path.exists(file_path):
        st.warning("No FAISS index found. Please process URLs to create one first.")
        return None

    try:
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        st.write(f"✅ FAISS index loaded from: {os.path.abspath(file_path)}")
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None


def query_index(query, vectorstore):
    """
    Queries the FAISS vector store using an LLM and returns the answer with sources.

    Parameters:
        query (str): User question/query.
        vectorstore (FAISS): Loaded FAISS vector store.

    Returns:
        dict or None: Dictionary containing 'answer' and 'sources' or None on error.
    """
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        return result
    except Exception as e:
        st.error(f"Error during query processing: {e}")
        return None


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("📰 News Research Tool 📈")
st.sidebar.title("🔗 Add News Article URLs")

# Sidebar input: Number of URLs
url_count = st.sidebar.number_input("Number of URLs to input", min_value=1, max_value=10, value=3, step=1)

# Sidebar input: URLs themselves
url_list = []
for i in range(url_count):
    url_input = st.sidebar.text_input(f"URL {i+1}")
    if url_input:
        url_list.append(url_input.strip())

# Sidebar buttons for processing/loading
process_clicked = st.sidebar.button("🚀 Process URLs and Create Index")
load_clicked = st.sidebar.button("📂 Load Existing Index")

# Process URLs button clicked
if process_clicked:
    if not url_list:
        st.warning("Please enter at least one valid URL before processing.")
    else:
        with st.spinner("Loading URLs and creating FAISS index..."):
            docs = load_data_from_urls(url_list)
            if docs:
                splitted_docs = split_documents(docs)
                vectorstore = create_faiss_index(splitted_docs)
                if vectorstore:
                    st.success("✅ URLs processed and FAISS index created successfully!")

# Load existing FAISS index button clicked
if load_clicked:
    vectorstore = load_faiss_index()
    if vectorstore:
        st.success("✅ FAISS index loaded successfully!")
    else:
        st.warning("No FAISS index found. Please process URLs first.")

# Main input: User question/query
query = st.text_input("🔍 Ask a question based on the news articles:")

# When user submits a query
if query:
    vectorstore = load_faiss_index()
    if vectorstore:
        with st.spinner("Searching for the answer..."):
            result = query_index(query, vectorstore)
            if result:
                st.header("📌 Answer")
                st.write(result.get("answer", "No answer found."))

                sources = result.get("sources", "")
                if sources:
                    st.subheader("📚 Sources")
                    # Show each source cleanly as a list
                    sources_list = [src.strip() for src in sources.split(',') if src.strip()]
                    for src in sources_list:
                        st.write(f"- {src}")
            else:
                st.error("Failed to retrieve an answer from the index.")
