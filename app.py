import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq

DEFAULT_ORDERS_FILE = "orders.xlsx"  # looked up next to app.py if the user doesn't upload one


def get_default_api_key():
    """Check Streamlit secrets first, then environment variable."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")

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
    :root {
        --bg-primary: #0b1220;
        --bg-secondary: #131c2e;
        --bg-card: #1a2536;
        --border-color: #2a3a52;
        --text-primary: #f1f5f9;
        --text-secondary: #a0aec0;
        --accent: #38bdf8;
        --accent-secondary: #a78bfa;
    }

    /* ---- BACKGROUND ---- */
    .stApp {
        background: linear-gradient(-45deg, var(--bg-primary), var(--bg-secondary), #1e2a45, var(--bg-primary));
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---- BASE TEXT: every element defaults to text-primary on the dark bg ---- */
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
    .stMarkdown, .stCaption {
        color: var(--text-primary) !important;
    }
    h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }

    /* ---- HEADER ---- */
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0;
        background: linear-gradient(90deg, var(--accent), var(--accent-secondary), var(--accent));
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shine 6s linear infinite;
    }
    @keyframes shine { to { background-position: 200% center; } }

    .sub-header {
        color: var(--text-secondary) !important;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
        opacity: 0;
        animation: fadeIn 1s ease forwards 0.3s;
    }

    /* ---- METRIC CARDS ---- */
    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1rem;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.25);
    }
    div[data-testid="stMetric"] label { color: var(--text-secondary) !important; }
    div[data-testid="stMetricValue"] { color: var(--text-primary) !important; }

    /* ---- CHAT ---- */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        animation: slideIn 0.35s ease;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] div {
        color: var(--text-primary) !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder { color: var(--text-secondary) !important; }
    div[data-testid="stChatInput"] textarea:focus {
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.5) !important;
    }

    /* ---- SIDEBAR ---- */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color);
    }
    section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }

    /* ---- INPUTS (main area) ---- */
    .stTextInput input, .stTextArea textarea,
    div[data-baseweb="select"] > div {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    .stTextInput input::placeholder { color: var(--text-secondary) !important; }

    /* ---- BUTTONS ---- */
    .stButton > button {
        background: var(--bg-card);
        color: var(--text-primary) !important;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: var(--accent);
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }

    /* ---- TABS ---- */
    button[data-baseweb="tab"] { font-weight: 600; transition: color 0.2s ease; }
    button[data-baseweb="tab"] p { color: var(--text-secondary) !important; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: var(--accent) !important; }

    /* ---- EXPANDERS ---- */
    .streamlit-expanderHeader, details summary { color: var(--text-primary) !important; }
    details {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px;
    }

    /* ---- DATAFRAME: keep its own light surface, force dark readable text on it ---- */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        animation: fadeIn 0.5s ease;
        background: #ffffff;
    }

    /* ---- ALERT BOXES (info/warning/error/success) ---- */
    div[data-testid="stAlert"] { border-radius: 10px; }
    div[data-testid="stAlert"] p { color: #0f172a !important; }

    /* ---- SCROLLBAR ---- */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(56, 189, 248, 0.4); border-radius: 4px; }

    /* =========================================================
       RESPONSIVE BREAKPOINTS
       ========================================================= */

    /* Base: tighten default padding on small screens so content
       isn't cramped against the edges */
    .block-container {
        padding-top: 1.5rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 1200px;
    }

    /* ---- TABLET (<= 992px) ---- */
    @media (max-width: 992px) {
        .main-header { font-size: 2.1rem; }
        .sub-header { font-size: 0.95rem; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }

    /* ---- MOBILE (<= 640px) ---- */
    @media (max-width: 640px) {
        .main-header { font-size: 1.6rem; }
        .sub-header { font-size: 0.85rem; margin-bottom: 1rem; }

        .block-container {
            padding-top: 1rem;
            padding-left: 0.6rem;
            padding-right: 0.6rem;
        }

        /* Metric cards: smaller padding + font so 4-in-a-row still fits */
        div[data-testid="stMetric"] {
            padding: 0.5rem;
            border-radius: 10px;
        }
        div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
        div[data-testid="stMetric"] label { font-size: 0.7rem !important; }

        /* Chat bubbles take full width, tighter padding */
        div[data-testid="stChatMessage"] {
            padding: 0.6rem !important;
            margin-bottom: 0.4rem;
        }

        /* Tabs: smaller text so both tabs fit without wrapping */
        button[data-baseweb="tab"] p { font-size: 0.85rem !important; }

        /* Buttons full-width friendly */
        .stButton > button { width: 100%; }

        /* Dataframe: allow horizontal scroll instead of squishing columns */
        div[data-testid="stDataFrame"] {
            overflow-x: auto !important;
        }
    }

    /* ---- VERY SMALL (<= 400px) ---- */
    @media (max-width: 400px) {
        .main-header { font-size: 1.35rem; }
        div[data-testid="stMetricValue"] { font-size: 0.95rem !important; }
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

    default_key = get_default_api_key()
    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Loaded automatically from st.secrets or the GROQ_API_KEY environment variable, if set."
    )
    if default_key:
        st.caption("✅ API key auto-loaded")

    uploaded_file = st.file_uploader(
        "Upload orders.xlsx (optional — uses the repo's orders.xlsx by default)",
        type=["xlsx"]
    )
    if uploaded_file is None and os.path.exists(DEFAULT_ORDERS_FILE):
        st.caption(f"✅ Using default: {DEFAULT_ORDERS_FILE}")

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
data_source = uploaded_file if uploaded_file is not None else (
    DEFAULT_ORDERS_FILE if os.path.exists(DEFAULT_ORDERS_FILE) else None
)

if data_source and groq_api_key:
    df = load_data(data_source)
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

        styler = display_df.style
        if hasattr(styler, "map"):
            styled = styler.map(style_status, subset=["Status"])
        else:
            styled = styler.applymap(style_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(display_df)} of {len(df)} orders")

else:
    missing = []
    if not groq_api_key:
        missing.append("a Groq API key (set `GROQ_API_KEY` as an env var / secret, or paste it in the sidebar)")
    if not data_source:
        missing.append(f"an orders file (place `{DEFAULT_ORDERS_FILE}` next to app.py, or upload one)")
    st.info("👈 Still needed: " + " and ".join(missing))
    st.markdown("""
    **What this app does:**
    1. Reads your order data from Excel
    2. Finds the most relevant rows for your question (TF-IDF search)
    3. Sends those rows to a Groq-hosted LLM to generate a plain-English answer
    """)
