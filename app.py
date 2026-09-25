import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Orders Q&A Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    /* Animated gradient background for the whole app */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e293b, #312e81, #0f172a);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Animated gradient text for the title */
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0;
        background: linear-gradient(90deg, #38bdf8, #a78bfa, #38bdf8);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shine 6s linear infinite;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }

    .sub-header {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
        opacity: 0;
        animation: fadeIn 1s ease forwards 0.3s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Glassmorphism cards for metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 1rem;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.25);
    }

    /* Chat message bubbles fade+slide in */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        animation: slideIn 0.35s ease;
        margin-bottom: 0.5rem;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #38bdf8;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
        transition: color 0.2s ease;
    }

    /* Dataframe container */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        animation: fadeIn 0.5s ease;
    }

    /* Chat input glow on focus */
    div[data-testid="stChatInput"] textarea:focus {
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.5) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(56, 189, 248, 0.4); border-radius: 4px; }

    /* ---- TEXT CONTRAST FIXES ---- */
    /* Force readable light text everywhere on the dark background */
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span,
    .stMarkdown, .stCaption, .stText {
        color: #e5e7eb !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
    }

    /* Metric labels and values */
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Input fields, text areas, selects - dark bg + light text */
    .stTextInput input, .stTextArea textarea,
    div[data-baseweb="select"] > div, div[data-testid="stChatInput"] textarea {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    .stTextInput input::placeholder, div[data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
    }

    /* Expander headers */
    .streamlit-expanderHeader, details summary {
        color: #e5e7eb !important;
    }

    /* Tabs text */
    button[data-baseweb="tab"] p {
        color: #cbd5e1 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #38bdf8 !important;
    }

    /* Chat message text */
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] div {
        color: #f1f5f9 !important;
    }

    /* Keep dataframe (rendered in its own light iframe) readable regardless of theme */
    div[data-testid="stDataFrame"] {
        background: #ffffff;
    }

    /* Info/warning/error boxes: keep default backgrounds but ensure text is dark enough on them */
    div[data-testid="stAlert"] p {
        color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)

STATUS_COLORS = {
    "Delivered": "#16a34a",
    "Shipped": "#2563eb",
    "Processing": "#d97706",
    "Pending": "#6b7280",
    "Cancelled": "#dc2626",
    "Returned": "#7c3aed",
}


def style_status(val):
    color = STATUS_COLORS.get(val, "#374151")
    return f"background-color: {color}20; color: {color}; font-weight: 600;"


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get one at console.groq.com/keys")
    uploaded_file = st.file_uploader("Upload orders.xlsx", type=["xlsx"])

    with st.expander("Advanced options"):
        model_name = st.selectbox(
            "Model",
            ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            index=0
        )
        top_k = st.slider("Rows retrieved as context", 1, 20, 5)

    st.markdown("---")
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with Streamlit + Groq + TF-IDF search")


# ---------------- DATA HELPERS ----------------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    def row_to_text(row):
        return (f"Order {row['Order ID']} placed by {row['Customer Name']} "
                f"for {row['Product']}, shipped to {row['Shipping Address']} "
                f"on {row['Order Date']}. Status: {row['Status']}.")

    df["text"] = df.apply(row_to_text, axis=1)
    return df


@st.cache_resource
def build_index(_df):
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(_df["text"])
    return vectorizer, doc_vectors


def search(query, df, vectorizer, doc_vectors, k=5):
    q_vec = vectorizer.transform([query])
    scores = (doc_vectors @ q_vec.T).toarray().flatten()
    top_idx = scores.argsort()[-k:][::-1]
    return df.iloc[top_idx]


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------- HEADER ----------------
st.markdown('<p class="main-header">📦 Orders Q&A Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your orders in plain English — powered by Groq</p>', unsafe_allow_html=True)

# ---------------- MAIN ----------------
if uploaded_file and groq_api_key:
    df = load_data(uploaded_file)
    vectorizer, doc_vectors = build_index(df)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders", len(df))
    with col2:
        st.metric("Delivered", int((df["Status"] == "Delivered").sum()))
    with col3:
        st.metric("Pending", int((df["Status"] == "Pending").sum()))
    with col4:
        st.metric("Cancelled", int((df["Status"] == "Cancelled").sum()))

    tab_chat, tab_data = st.tabs(["💬 Chat", "📋 Order Data"])

    # ---------------- CHAT TAB ----------------
    with tab_chat:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and "context_df" in msg:
                    with st.expander("Sources used"):
                        st.dataframe(msg["context_df"], use_container_width=True, hide_index=True)

        query = st.chat_input("Ask about your orders... e.g. 'which orders are cancelled?'")

        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching orders and thinking..."):
                    results = search(query, df, vectorizer, doc_vectors, k=top_k)
                    context = "\n".join(results["text"].tolist())

                    try:
                        client = Groq(api_key=groq_api_key)
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "Answer strictly based on the order data provided. If the answer isn't in the data, say so clearly."},
                                {"role": "user", "content": f"Order data:\n{context}\n\nQuestion: {query}"}
                            ]
                        )
                        answer = response.choices[0].message.content
                    except Exception as e:
                        answer = f"⚠️ Error calling Groq API: {e}"

                st.write(answer)
                context_display = results.drop(columns=["text"])
                with st.expander("Sources used"):
                    st.dataframe(context_display, use_container_width=True, hide_index=True)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "context_df": results.drop(columns=["text"])
            })

    # ---------------- DATA TAB ----------------
    with tab_data:
        st.markdown("#### Full order dataset")
        search_box = st.text_input("🔍 Filter by keyword (any column)")
        display_df = df.drop(columns=["text"])
        if search_box:
            mask = display_df.apply(lambda r: r.astype(str).str.contains(search_box, case=False).any(), axis=1)
            display_df = display_df[mask]

        styled = display_df.style.applymap(style_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(display_df)} of {len(df)} orders")

else:
    st.info("👈 Enter your Groq API key and upload `orders.xlsx` in the sidebar to get started.")
    st.markdown("""
    **What this app does:**
    1. Reads your order data from Excel
    2. Finds the most relevant rows for your question (TF-IDF search)
    3. Sends those rows to a Groq-hosted LLM to generate a plain-English answer
    """)
