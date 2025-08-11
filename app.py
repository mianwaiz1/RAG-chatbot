import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from gtts import gTTS
import tempfile
import numpy as np
import fitz

load_dotenv()

# ---------- PAGE COUNT ----------
def get_pdf_page_count(path):
    pdf = fitz.open(path)
    return pdf.page_count

# ---------- SETUP GEMINI ----------
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
os.environ["GOOGLE_API_KEY"] = google_api_key

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ---------- AUTO PARAMS ----------
def get_dynamic_params(page_count: int):
    if page_count <= 50:
        return 600, 100, 4
    elif page_count <= 200:
        return 800, 150, 6
    elif page_count <= 500:
        return 1000, 150, 8
    else:
        return 1000, 200, 8

# ---------- LOADER, SPLITTER, EMBEDDER ----------
def build_vectorstore_from_path(path):
    page_count = get_pdf_page_count(path)
    chunk_size, chunk_overlap, _ = get_dynamic_params(page_count)  # ignore top_k here

    loader = PyMuPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding=embedder)
    return vectorstore, [doc.page_content for doc in chunks], page_count

# ---------- RAG + Scoring ----------
def get_answer(query, vectorstore, chunks, page_count):
    _, _, top_k = get_dynamic_params(page_count)
    docs = vectorstore.similarity_search(query, k=top_k)
    references = [doc.page_content for doc in docs]
    context = "\n\n".join(references)

    # Include chat memory (last 10 interactions)
    chat_history = ""
    for msg in st.session_state.messages[-10:]:
        if msg["role"] == "user":
            chat_history += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            chat_history += f"Assistant: {msg['content']}\n"

    prompt = f"""
Use the following document context and conversation history to answer the question as accurately as possible.

If the answer is not mentioned in the context, respond with:
"Not mentioned in the document."

--- Conversation History ---
{chat_history}

--- Document Context ---
{context}

--- Question ---
{query}
"""

    answer = llm.invoke(prompt).content
    score = semantic_score_google_embeddings(answer, references)
    return answer.strip(), references, score#type: ignore

def semantic_score_google_embeddings(answer, references):
    try:
        ans_emb = embedder.embed_query(answer)
        ref_embs = [embedder.embed_query(ref) for ref in references]

        similarities = [
            np.dot(ans_emb, ref_emb) / (np.linalg.norm(ans_emb) * np.linalg.norm(ref_emb))
            for ref_emb in ref_embs
        ]

        # Take the average of top 3 similarities for stable scoring
        top_similarities = sorted(similarities, reverse=True)[:3]
        return round(float(np.mean(top_similarities)) * 100, 2)
    except Exception as e:
        print("Semantic scoring failed:", e)
        return 0.0


# ---------- STREAMLIT UI ----------
st.set_page_config("📄 RAG Chatbot", layout="wide")
st.title("🧠 RAG Chatbot PDF")
with st.expander("How to use..."):
    st.markdown("Upload any PDF to ask questions based on its content.")
    st.markdown("The accuracy depends on the relation between context of document and the question.")
    st.markdown("Snippets has some context of document.")
st.markdown("---")

uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    with st.spinner("Processing uploaded PDF..."):
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        vectorstore, chunks, page_count = build_vectorstore_from_path("uploaded.pdf")

        chunk_size, chunk_overlap, top_k = get_dynamic_params(page_count)

        st.session_state.vectorstore = vectorstore
        st.session_state.chunks = chunks
        st.session_state.page_count = page_count
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap
        st.session_state.top_k = top_k

        st.success(f"✅ PDF processed! Page count: {page_count}. You can now ask questions.")
        
if "vectorstore" not in st.session_state or "chunks" not in st.session_state or "page_count" not in st.session_state:
    st.warning("📄 Please upload a PDF first to enable the chatbot.")
    st.stop()

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
            st.caption("The accuracy depends on the relation between context of document and the question")
            st.markdown(f"📊 **Answer Accuracy: {acc}%** ({color})")
            

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()
with st.sidebar.expander("More Info..."):
    st.markdown("models: Gimini-2.0-flash")
    st.markdown("embeder: GoogleGenerativeAIEmbeddings(model=embedding-001)")
    st.markdown("token usage: 80.78k")
    if "page_count" in st.session_state:
        st.caption(
            f"📄 Pages: {st.session_state.page_count} | "
            f"🔹 Chunk Size: {st.session_state.chunk_size} | "
            f"🔹 Overlap: {st.session_state.chunk_overlap} | "
            f"🔹 top_k: {st.session_state.top_k}"
    )
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
                st.session_state.chunks,
                st.session_state.page_count
            )
            st.markdown(answer)

            if accuracy > 80:
                label = "🟢 High"
            elif accuracy > 50:
                label = "🟡 Medium"
            else:
                label = "🔴 Low"
            st.markdown(f"📊 **Answer Accuracy: {accuracy}%** ({label})")
            tts = gTTS(answer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
                tmp_path = tmpfile.name
            tts.save(tmp_path)

            with open(tmp_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')
            os.remove(tmp_path)
            with st.expander("📚 Source Snippets"):
                for i, snippet in enumerate(sources):
                    st.markdown(f"**Snippet {i+1}:**\n> {snippet[:300]}...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "accuracy": accuracy
    })

