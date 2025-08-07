import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
import numpy as np

load_dotenv()

# ---------- SETUP GEMINI ----------
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
os.environ["GOOGLE_API_KEY"] = google_api_key

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ---------- LOADER, SPLITTER, EMBEDDER ----------
def build_vectorstore_from_path(path):
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding=embedder)
    return vectorstore, [doc.page_content for doc in chunks]

# ---------- RAG + Scoring ----------
def get_answer(query, vectorstore, chunks, top_k=3):
    docs = vectorstore.similarity_search(query, k=top_k)
    references = [doc.page_content for doc in docs]
    context = "\n\n".join(references)

    prompt = f"""Answer the question using only the context below:

Context:
{context}

Question:
{query}"""

    answer = llm.invoke(prompt).content
    score = semantic_score_google_embeddings(answer, references)
    return answer, references, score

def semantic_score_google_embeddings(answer, references):
    try:
        ans_emb = embedder.embed_query(answer)
        ref_embs = [embedder.embed_query(ref) for ref in references]

        # Calculate cosine similarity
        similarities = [np.dot(ans_emb, ref_emb) / (np.linalg.norm(ans_emb) * np.linalg.norm(ref_emb)) for ref_emb in ref_embs]
        return round(float(max(similarities)) * 100, 2)
    except Exception as e:
        print("Semantic scoring failed:", e)
        return 0.0

# ---------- STREAMLIT UI ----------
st.set_page_config("📄 RAG Chatbot", layout="wide")
st.title("🧠 RAG Chatbot (Gemini + Google Embeddings)")
st.caption("Upload any PDF to ask questions based on its content.")
st.markdown("---")

uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
use_custom_pdf = uploaded_file is not None

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load vectorstore
if uploaded_file:
    # Handle uploaded file and set session_state.vectorstore
    with st.spinner("Processing uploaded PDF..."):
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        vectorstore, chunks = build_vectorstore_from_path("uploaded.pdf")
        st.session_state.vectorstore = vectorstore
        st.session_state.chunks = chunks

        st.success("✅ PDF processed! You can now ask questions.")

# ✅ Prevent chat until vectorstore is ready
if "vectorstore" not in st.session_state or "chunks" not in st.session_state:
    st.warning("📄 Please upload a PDF first to enable the chatbot.")
    st.stop()  # ⛔ Stop the app from running further


# Show chat history
for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if "accuracy" in entry:
            acc = entry["accuracy"]
            if acc > 80:
                color = "🟢 High"
            elif acc > 50:
                color = "🟡 Medium"
            else:
                color = "🔴 Low"
            st.markdown(f"📊 **Answer Accuracy: {acc}%** ({color})")

# Chat input
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()
query = st.chat_input("Ask a question...")
if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources, accuracy = get_answer(
                query,
                st.session_state.vectorstore,
                st.session_state.chunks
            )
            st.markdown(answer)
            st.caption(f"📊 Answer Accuracy: **{accuracy}%**")
            with st.expander("📚 Source Snippets"):
                for i, snippet in enumerate(sources):
                    st.markdown(f"**Snippet {i+1}:**\n> {snippet[:300]}...")    
    if st.download_button("💾 Download Chat History", 
        data="\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages]),
        file_name="chat_history.txt",
        use_container_width=True):
        st.success("✅ Chat downloaded!")
        
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "accuracy": accuracy
    })

