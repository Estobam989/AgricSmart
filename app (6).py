
import sys
import os
# Add the directory of the current script to sys.path for local module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import os
import tempfile
import shutil
import config # Import config

# Try to import necessary libraries, install if missing
try:
    from langchain_groq import ChatGroq
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "langchain-groq"])
    from langchain_groq import ChatGroq

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "langchain-huggingface"])
    from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "langchain-community", "pypdf", "duckduckgo-search"])
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.tools import DuckDuckGoSearchRun

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "langchain-text-splitters"])
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document


# --- 1. Environment Variables ---
# These are now loaded from config.py
GROQ_API_KEY = config.GROQ_API_KEY
MODEL_NAME = config.MODEL_NAME
TEMPERATURE = config.TEMPERATURE
MAX_TOKENS = config.MAX_TOKENS

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")
    st.stop()

# --- 2. LLM Initialization ---
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)

# --- 3. Embeddings Initialization ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 4. Define Tools ---
# Farm Profit Calculator
def calculate_profit(cost, revenue):
    return revenue - cost

# Web Search Tool
search = DuckDuckGoSearchRun()

# --- 5. Document Loading, Splitting, and Vector Store (RAG Setup) ---
VECTOR_STORE_PATH = "faiss_index"
VECTOR_STORE_NAME = "index"

# Initialize vectorstore and retriever
vectorstore = None
retriever = None

# Streamlit UI for Sidebar and PDF upload
with st.sidebar:
    st.title("🌾 AgriSmart AI")
    st.markdown("---")
    st.write("✔ AI Chat")
    st.write("✔ PDF Knowledge")
    st.write("✔ Calculator")
    st.write("✔ Web Search")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Agricultural PDF",
        type="pdf"
    )

documents = []
docs = []

if uploaded_file is not None:
    # Save uploaded file to a temporary location for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        st.sidebar.success(f"Loaded {len(documents)} pages from uploaded PDF.")
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded PDF: {e}")
    finally:
        os.remove(temp_file_path) # Clean up temporary file

if documents: # If documents were loaded from uploaded PDF
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    if docs:
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        vectorstore.save_local(folder_path=VECTOR_STORE_PATH, index_name=VECTOR_STORE_NAME)
        st.sidebar.success("Vector store created from uploaded PDF.")
        retriever = vectorstore.as_retriever()
    else:
        st.sidebar.warning("No chunks created from uploaded PDF.")
elif os.path.exists(VECTOR_STORE_PATH) and os.path.isdir(VECTOR_STORE_PATH): # Load existing vector store
    try:
        vectorstore = FAISS.load_local(
            folder_path=VECTOR_STORE_PATH,
            index_name=VECTOR_STORE_NAME,
            embeddings=embeddings,
            allow_dangerous_deserialization=True # Required for FAISS.load_local with custom embeddings
        )
        st.sidebar.info("Loaded existing FAISS vector store.")
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.sidebar.error(f"Error loading existing FAISS vector store: {e}")
else: # Fallback to mock document if no PDF uploaded and no existing vector store
    mock_content = """
Agriculture is the backbone of many economies, providing food, fiber, and raw materials.
Sustainable agriculture practices are crucial for long-term ecological balance and food security.
These practices include crop rotation, organic farming, water conservation, and reduced pesticide use.
Precision agriculture, leveraging technologies like IoT and AI, is transforming farming by optimizing resource allocation and improving yields.
Key crops include wheat, corn, rice, and soybeans, which are staple foods globally.
Livestock farming also plays a significant role, contributing to dairy, meat, and wool production.
Challenges in agriculture include climate change, soil degradation, and pest resistance, necessitating continuous innovation and research.
    """
    mock_doc = Document(page_content=mock_content, metadata={'source': 'mock_document.txt', 'page': 1})
    documents = [mock_doc]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    if docs:
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        vectorstore.save_local(folder_path=VECTOR_STORE_PATH, index_name=VECTOR_STORE_NAME)
        st.sidebar.info("Using mock data to create vector store as no PDF was uploaded or or existing vector store was found.")
        retriever = vectorstore.as_retriever()
    else:
        st.sidebar.error("Failed to create vector store even with mock data.")

# --- 6. Conversation Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 7. Chat Interface ---
st.title("🌱 AgriSmart AI")
prompt = st.chat_input("Ask me anything about farming...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Tool Routing Logic
    response_content = ""
    if "price" in prompt.lower():
        response_content = search.run(prompt)
    elif "profit" in prompt.lower():
        # Example hardcoded profit calculation, adjust inputs as needed
        response_content = f"The estimated profit is: ${calculate_profit(500000, 850000):,.2f}"
    elif retriever and ("agriculture" in prompt.lower() or "farm" in prompt.lower() or "crop" in prompt.lower() or "soil" in prompt.lower() or "irrigation" in prompt.lower()):
        # Use RAG if retriever is available and prompt is related to agricultural topics
        retrieved_docs = retriever.invoke(prompt)
        context = " ".join([d.page_content for d in retrieved_docs])
        # A simple prompt template for RAG
        rag_prompt = f"Based on the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        response_content = llm.invoke(rag_prompt).content
    else:
        # Default to LLM without specific tools/RAG
        response_content = llm.invoke(prompt).content

    st.session_state.messages.append({"role": "assistant", "content": response_content})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
