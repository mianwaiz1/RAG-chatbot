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
import numpy as np
from gtts import gTTS
import tempfile

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
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding=embedder)
    return vectorstore, [doc.page_content for doc in chunks]

# ---------- RAG + Scoring ----------
def get_answer(query, vectorstore, chunks, top_k=8):
    docs = vectorstore.similarity_search(query, k=top_k)
    references = [doc.page_content for doc in docs]
    context = "\n\n".join(references)

    # Include chat memory (last 5 interactions)
    chat_history = ""
    for msg in st.session_state.messages[-10:]:  # adjust as needed
        if msg["role"] == "user":
            chat_history += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            chat_history += f"Assistant: {msg['content']}\n"

    prompt = f"""
Use the following document context and conversation history to answer the question as accurately as possible.

If the answer is not mentioned in the context, respond with:
"Not mentioned in the document. However, based on my knowledge: ..."

--- Conversation History ---
{chat_history}

--- Document Context ---
{context}

--- Question ---
{query}
"""

    answer = llm.invoke(prompt).content
    score = semantic_score_google_embeddings(answer, references)
    return answer.strip(), references, score

def semantic_score_google_embeddings(answer, references):
    try:
        ans_emb = embedder.embed_query(answer)
        ref_embs = [embedder.embed_query(ref) for ref in references]

        similarities = [
            np.dot(ans_emb, ref_emb) / 
            (np.linalg.norm(ans_emb) * np.linalg.norm(ref_emb))
            for ref_emb in ref_embs
        ]

        # Take the average of top 3 similarities for more stable scoring
        top_similarities = sorted(similarities, reverse=True)[:3]
        return round(float(np.mean(top_similarities)) * 100, 2)
    except Exception as e:
        print("Semantic scoring failed:", e)
        return 0.0


# ---------- STREAMLIT UI ----------
st.set_page_config("📄 RAG Chatbot", layout="wide")
st.title("🧠 RAG Chatbot PDF")
st.caption("Upload any PDF to ask questions based on its content.")
st.markdown("---")

uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
use_custom_pdf = uploaded_file is not None

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    with st.spinner("Processing uploaded PDF..."):
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        vectorstore, chunks = build_vectorstore_from_path("uploaded.pdf")
        st.session_state.vectorstore = vectorstore
        st.session_state.chunks = chunks
        st.success("✅ PDF processed! You can now ask questions.")

if "vectorstore" not in st.session_state or "chunks" not in st.session_state:
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
            st.caption("The accuracy is depending on the relation between context of document and the question")
            st.markdown(f"📊 **Answer Accuracy: {acc}%** ({color})")

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

            # Show accuracy score with color label
            if accuracy > 80:
                label = "🟢 High"
            elif accuracy > 50:
                label = "🟡 Medium"
            else:
                label = "🔴 Low"

            st.markdown(f"📊 **Answer Accuracy: {accuracy}%** ({label})")


            # ✅ TEXT TO SPEECH SECTION
            try:
                tts = gTTS(answer)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
                    tts.save(tmpfile.name)
                    audio_bytes = open(tmpfile.name, 'rb').read()
                    st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                st.warning(f"TTS failed: {e}")

    
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






