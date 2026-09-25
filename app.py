import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq

st.set_page_config(page_title="Orders Q&A (Groq)", layout="wide")
st.title("📦 Orders Q&A — powered by Groq")

# --- Sidebar: API key + file upload ---
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload orders.xlsx", type=["xlsx"])
top_k = st.sidebar.slider("Rows to retrieve as context", 1, 20, 5)
model_name = st.sidebar.selectbox(
    "Groq model",
    ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    index=0
)


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


if uploaded_file and groq_api_key:
    df = load_data(uploaded_file)
    vectorizer, doc_vectors = build_index(df)

    st.success(f"Loaded {len(df)} orders.")
    st.dataframe(df.drop(columns=["text"]), use_container_width=True)

    query = st.text_input("Ask a question about your orders:")

    if st.button("Ask") and query:
        with st.spinner("Searching and thinking..."):
            results = search(query, df, vectorizer, doc_vectors, k=top_k)
            context = "\n".join(results["text"].tolist())

            try:
                client = Groq(api_key=groq_api_key)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Answer strictly based on the order data provided. If the answer isn't in the data, say so."},
                        {"role": "user", "content": f"Order data:\n{context}\n\nQuestion: {query}"}
                    ]
                )
                answer = response.choices[0].message.content
            except Exception as e:
                st.error(f"Groq API error: {e}")
                answer = None

        if answer:
            st.subheader("Answer")
            st.write(answer)

            with st.expander("Matched order rows used as context"):
                st.dataframe(results.drop(columns=["text"]), use_container_width=True)
else:
    st.info("Enter your Groq API key and upload orders.xlsx to get started.")
