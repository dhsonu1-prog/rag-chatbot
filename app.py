import streamlit as st
import os
import sys
import re

# Add current directory to path so query_engine can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query_engine import query_rag

# Constants
workspace_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(workspace_dir, "qdrant_db")

# Page configuration
st.set_page_config(
    page_title="Offline Civil Services RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Dark mode background with slight gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f1f5f9;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0f172a;
    }
    
    /* Custom container/glassmorphism cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Chat message styling overrides */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    
    .stChatMessage[data-testid="chatAvatarIcon-user"] {
        background-color: #38bdf8;
    }
    
    .stChatMessage[data-testid="chatAvatarIcon-assistant"] {
        background-color: #a855f7;
    }
</style>
""", unsafe_allow_html=True)

# Helper to format source citations and extract dates
def format_source_citation(filepath):
    filename = os.path.basename(filepath)
    # Extract date if file contains "dated <date>"
    date_match = re.search(r"dated\s+([A-Za-z0-9\s_-]+)", filename, re.IGNORECASE)
    date_str = ""
    if date_match:
        raw_date = date_match.group(1).strip()
        raw_date = re.sub(r"\.(pdf|docx|html|txt|png|jpg|jpeg|tiff)$", "", raw_date, flags=re.IGNORECASE)
        date_str = f" ({raw_date})"
    
    # Extract parent directory category
    parent_dir = os.path.basename(os.path.dirname(filepath))
    category = f"[{parent_dir}] " if parent_dir and parent_dir not in ["docs", "Downloads"] else ""
    
    # Strip date suffix and extension
    display_name = re.sub(r"\s+dated\s+.*", "", filename, flags=re.IGNORECASE)
    display_name = re.sub(r"\.(pdf|png|jpg|jpeg|tiff)$", "", display_name, flags=re.IGNORECASE)
    
    return f"📄 {category}**{display_name}**{date_str}"

# App state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header UI
st.markdown('<div class="main-title">📚 Local RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Search and consult 115 Civil Services OMs, Rules, & Guidelines 100% offline.</div>', unsafe_allow_html=True)

# Sidebar configurations
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/database.png", width=80)
    st.markdown("### ⚙️ Pipeline Settings")
    
    # LLM Source configuration
    llm_source = st.radio(
        "Select LLM Source",
        ["Local Ollama", "Remote vLLM Server"],
        index=0,
        help="Use a local Ollama model or point to the remote vLLM model at 172.16.172.4:3003"
    )
    
    api_base = None
    api_model = None
    model_choice = "gemma2:9b"
    
    if llm_source == "Local Ollama":
        model_choice = st.selectbox(
            "Choose Local LLM (Ollama)",
            ["gemma2:9b", "gemma2:2b", "llama3.1:8b", "qwen2:7b", "llama3"],
            index=0,
            help="Make sure you have pulled this model locally using: 'ollama pull <model_name>'"
        )
    else:
        api_base = st.text_input("vLLM Base URL", value="http://172.16.172.4:3003/v1/")
        api_model = st.text_input("Model ID", value="/mnt/ai_storage/models/Qwen3.5-397B-A17B-FP8-dynamic")
    
    st.markdown("### 🎛️ Hybrid Search (RRF)")
    vector_weight = st.slider(
        "Vector (Semantic) Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher values focus more on semantic meaning. Lower values focus more on exact keyword matching."
    )
    bm25_weight = round(1.0 - vector_weight, 2)
    st.caption(f"BM25 (Keyword) Weight: **{bm25_weight}**")
    
    st.markdown("### 🚀 Reranker Settings")
    use_reranker = st.toggle("Enable Reranking (Qwen3-VL)", value=True, help="Use Qwen3-VL-Reranker-2B to rerank candidates on your GPU.")
    top_n_rerank = st.slider("Top N Documents to Keep", min_value=1, max_value=10, value=5, help="Number of reranked documents to send as context to the LLM.")
    
    st.markdown("---")
    st.markdown("### 📊 Database Status")
    
    # Check if Chroma DB directory exists
    db_exists = os.path.exists(db_dir)
    if db_exists:
        st.success("✅ Database loaded successfully")
        # Count files dynamically
        pdf_count = 0
        image_count = 0
        exclude_dirs = {'.git', 'venv', '.venv', 'qdrant_db', 'node_modules', '__pycache__'}
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff')
        for root, dirs, files in os.walk(workspace_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for f in files:
                ext = f.lower()
                if ext.endswith('.pdf'):
                    pdf_count += 1
                elif ext.endswith(image_extensions):
                    image_count += 1
            
        st.info(f"📁 PDFs: **{pdf_count}** | Images: **{image_count}**")
    else:
        st.warning("⚠️ Database not found")
        st.markdown("""
        Please build the database first by running:
        ```bash
        python ingest.py
        ```
        """)
        
    st.markdown("---")
    st.markdown("### 💡 Running Locally?")
    st.markdown("""
    Ensure **Ollama** is running locally and has the model:
    ```bash
    # Run Ollama server
    ollama run gemma2:2b
    ```
    Data stays private on your machine. No cloud API calls.
    """)
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Cited Sources"):
                # Limit to 5 sources max
                limited_sources = list(message["sources"].items())[:5]
                for src, pages in limited_sources:
                    pages_str = ", ".join(map(str, sorted(set(pages))))
                    st.markdown(f"{format_source_citation(src)} (Page {pages_str})")

# User Query input
if user_query := st.chat_input("Ask a question about Civil Services rules (e.g., reservation, pay scales, MACP)..."):
    
    # Check database presence
    if not db_exists:
        st.error("Chroma DB not found. Please run the Ingestion script first (`python ingest.py`).")
    else:
        # Display user question
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Build chat history for conversational RAG
        chat_history = []
        user_msg = None
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant" and user_msg is not None:
                chat_history.append((user_msg, msg["content"]))
                user_msg = None
                
        # Query local RAG pipeline with loader/spinner
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            with st.spinner("Searching local documents and generating response..."):
                answer, sources = query_rag(
                    user_query, 
                    model_name=model_choice, 
                    chat_history=chat_history,
                    api_base=api_base,
                    api_model=api_model,
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight,
                    use_reranker=use_reranker,
                    top_n_rerank=top_n_rerank
                )
                
            response_placeholder.markdown(answer)
            
            if sources:
                with st.expander("🔍 Cited Sources"):
                    # Limit to 5 sources max
                    limited_sources = list(sources.items())[:5]
                    for src, pages in limited_sources:
                        pages_str = ", ".join(map(str, sorted(set(pages))))
                        st.markdown(f"{format_source_citation(src)} (Page {pages_str})")
            
            # Save assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
