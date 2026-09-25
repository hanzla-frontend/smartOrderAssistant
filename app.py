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
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
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
