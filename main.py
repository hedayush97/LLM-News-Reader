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

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

INDEX_FILE = "faiss_index.pkl"

def fetch_text_from_url(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n\n".join(p.get_text() for p in paragraphs if p.get_text().strip())
        if not text.strip():
            st.warning(f"No textual content found at {url}")
        return Document(page_content=text, metadata={"source": url})
    except Exception as e:
        st.warning(f"Failed to fetch or parse URL {url}: {e}")
        return None

def load_data_from_urls(url_list):
    docs = []
    for url in url_list:
        doc = fetch_text_from_url(url)
        if doc:
            docs.append(doc)
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

def create_faiss_index(docs, file_path=INDEX_FILE):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    return vectorstore

def load_faiss_index(file_path=INDEX_FILE):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

def query_index(query, vectorstore):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    return chain({"question": query}, return_only_outputs=True)

# Streamlit UI
st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("📰 News Research Tool 📈")
st.sidebar.title("🔗 Add News Article URLs")

# URL inputs
url_count = st.sidebar.number_input("Number of URLs", min_value=1, max_value=10, value=3)
url_list = []
for i in range(url_count):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        url_list.append(url.strip())

# Session state for vectorstore and docs count
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

if st.sidebar.button("🚀 Process URLs"):
    if not url_list:
        st.warning("Please enter at least one valid URL.")
    else:
        with st.spinner("Loading URLs..."):
            docs = load_data_from_urls(url_list)
        st.success(f"Loaded {len(docs)} documents.")

        with st.spinner("Splitting documents into chunks..."):
            splitted_docs = split_documents(docs)
        st.success(f"Split documents into {len(splitted_docs)} chunks.")

        with st.spinner("Creating embeddings and FAISS index..."):
            vectorstore = create_faiss_index(splitted_docs)
        st.success("FAISS index created and saved!")

        st.session_state.vectorstore = vectorstore
        st.session_state.chunks_count = len(splitted_docs)

# Query input
query = st.text_input("🔍 Ask a question based on the news articles:")

if query:
    # Load index if not in session state
    if st.session_state.vectorstore is None:
        vectorstore = load_faiss_index()
        if vectorstore is None:
            st.error("No FAISS index found. Please process URLs first.")
        else:
            st.session_state.vectorstore = vectorstore

    if st.session_state.vectorstore:
        with st.spinner("Searching for the answer..."):
            result = query_index(query, st.session_state.vectorstore)

        if result:
            st.header("📌 Answer")
            st.write(result.get("answer", "No answer found."))

            sources = result.get("sources", "")
            if sources:
                st.subheader("📚 Sources")
                sources_list = [src.strip() for src in sources.split(",") if src.strip()]
                for src in sources_list:
                    st.write(f"- {src}")

# Optional: show how many chunks processed
if st.session_state.chunks_count > 0:
    st.sidebar.markdown(f"**Chunks created:** {st.session_state.chunks_count}")
