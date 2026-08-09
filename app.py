# here we use gemini for frontend

import streamlit as st
import config

from rag.vector_store import (
    calculate_folder_hash,
    load_hash,
    save_hash,
    build_vector_store,
    save_vector_store,
)

from rag.pipeline import generate_answer


# ==========================================================
# 1. Streamlit Page Configuration & Modern Theme Setup
# ==========================================================

st.set_page_config(
    page_title="Foodpanda Pakistan — RAG Assistant",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Foodpanda CSS Injection (Brand Styling, Hover Effects, Micro-Animations)
st.markdown("""
    <style>
    /* Main Theme Variables */
    :root {
        --brand-pink: #D70F64;
        --brand-pink-hover: #C00A56;
        --brand-pink-soft: rgba(215, 15, 100, 0.08);
    }

    /* Hide Default Streamlit Elements for Clean Look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Global Typography & Background Adjustments */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Custom Header Styling */
    .foodpanda-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 20px;
        background: white;
        border-bottom: 1px solid #E5E7EB;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Dark mode override for header */
    @media (prefers-color-scheme: dark) {
        .foodpanda-header {
            background: #1E1E1E;
            border-bottom: 1px solid #2D2D2D;
        }
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #ECFDF5;
        color: #059669;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
    }

    /* Custom Styled Buttons */
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton>button:hover {
        border-color: var(--brand-pink) !important;
        color: var(--brand-pink) !important;
        transform: translateY(-1px);
    }

    /* Primary Accent Pink Buttons */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: var(--brand-pink) !important;
        color: white !important;
        border: none !important;
    }
    
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: var(--brand-pink-hover) !important;
        color: white !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #E5E7EB;
    }

    /* Welcome Screen Card Enhancements */
    .suggested-card-box {
        border: 1px solid #E5E7EB;
        padding: 16px;
        border-radius: 12px;
        background: white;
        transition: all 0.2s ease;
        margin-bottom: 10px;
    }

    .suggested-card-box:hover {
        border-color: var(--brand-pink);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(215, 15, 100, 0.1);
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================================
# 2. Ensure FAISS Vector Store is Ready
# ==========================================================

@st.cache_resource
def ensure_index_ready():

    current_hash = calculate_folder_hash()
    previous_hash = load_hash()

    index_path = config.FAISS_INDEX_DIR / "index.faiss"
    documents_path = config.FAISS_INDEX_DIR / "documents.pkl"

    index_exists = index_path.exists()
    documents_exist = documents_path.exists()

    if (
        not index_exists
        or not documents_exist
        or current_hash != previous_hash
    ):
        with st.spinner("⚡ Building Knowledge Base Index... This may take a moment."):
            index, documents = build_vector_store()
            save_vector_store(index, documents)
            save_hash(current_hash)

    return True


# Prepare Vector Store
ensure_index_ready()


# ==========================================================
# 3. Sidebar Navigation & Metadata
# ==========================================================

with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="background: rgba(215, 15, 100, 0.1); padding: 10px; border-radius: 50%; font-size: 24px;">🐼</div>
            <div>
                <h3 style="margin: 0; font-size: 18px; font-weight: 700;">Foodpanda PK</h3>
                <p style="margin: 0; font-size: 12px; color: #6B7280;">AI Knowledge Assistant</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Quick Architecture Info Accordion
    with st.expander("ℹ️ About System Architecture"):
        st.markdown("""
        **Pipeline Overview:**
        - **Model:** `sentence-transformers/all-MiniLM-L6-v2`
        - **Vector DB:** FAISS Index
        - **LLM Core:** Gemini API
        - **Grounded:** 100% verified Foodpanda policy files.
        """)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="font-size: 11px; color: #9CA3AF; text-align: center;">
            Foodpanda Pakistan RAG Assistant<br>
            Powered by FAISS & Gemini
        </div>
    """, unsafe_allow_html=True)


# ==========================================================
# 4. App Main Header
# ==========================================================

st.markdown("""
    <div class="foodpanda-header">
        <div>
            <h2 style="margin: 0; font-size: 20px; font-weight: 700;">🐼 Foodpanda Pakistan Assistant</h2>
            <p style="margin: 4px 0 0 0; font-size: 12px; color: #6B7280;">
                Answers are grounded strictly in official knowledge base files.
            </p>
        </div>
        <div>
            <span class="status-badge">● Online</span>
        </div>
    </div>
""", unsafe_allow_html=True)


# ==========================================================
# 5. Session State & Messages
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================================
# 6. Welcome Screen (Displayed when no messages exist)
# ==========================================================

if len(st.session_state.messages) == 0:
    st.markdown("""
        <div style="text-align: center; padding: 30px 10px 10px 10px;">
            <div style="font-size: 48px; margin-bottom: 10px;">🐼</div>
            <h2 style="font-size: 24px; font-weight: 700; margin-bottom: 8px;">How can I help you today?</h2>
            <p style="font-size: 14px; color: #6B7280; max-width: 500px; margin: 0 auto 25px auto;">
                Ask me about Foodpanda Pakistan policies, payments, refunds, delivery, Pandapay, restaurants, and customer support.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Clickable Suggested Questions
    st.markdown("##### 💡 Suggested Questions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💳 What payment methods are available?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What payment methods are available?"})
            st.rerun()

        if st.button("📦 What if my order never arrived?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What should I do if my order never arrived?"})
            st.rerun()

    with col2:
        if st.button("💸 How can I get a refund for my order?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "How can I get a refund for my order?"})
            st.rerun()

        if st.button("👛 How does Pandapay wallet work?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "How does Pandapay wallet work?"})
            st.rerun()


# ==========================================================
# 7. Render Chat History
# ==========================================================

for message in st.session_state.messages:
    avatar = "🐼" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        # Display metadata sources if present in the message dict
        if "sources" in message and message["sources"]:
            with st.expander("📄 View Knowledge Base Sources"):
                for src in message["sources"]:
                    st.caption(f"• `{src}`")


# ==========================================================
# 8. User Input & Response Generation
# ==========================================================

question = st.chat_input("Ask a question about Foodpanda Pakistan...")

# Trigger generation if question typed OR if selected from suggested cards
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.rerun()

# Process latest pending user question
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    
    # Check if assistant already responded to avoid duplication
    if len(st.session_state.messages) % 2 != 0:
        latest_question = st.session_state.messages[-1]["content"]

        with st.chat_message("assistant", avatar="🐼"):
            with st.spinner("Searching knowledge base..."):
                try:
                    # Execute backend RAG pipeline
                    result = generate_answer(latest_question)

                    # Support both string response or tuple/dict response (answer + sources)
                    if isinstance(result, tuple):
                        answer, sources = result[0], result[1]
                    elif isinstance(result, dict):
                        answer = result.get("answer", "")
                        sources = result.get("sources", [])
                    else:
                        answer = result
                        sources = []

                    st.markdown(answer)

                    if sources:
                        with st.expander("📄 View Knowledge Base Sources"):
                            for src in sources:
                                st.caption(f"• `{src}`")

                except Exception as e:
                    answer = "I couldn't process that request right now. Please try again."
                    sources = []
                    st.error(answer)
                    st.exception(e)

        # Store response in session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
