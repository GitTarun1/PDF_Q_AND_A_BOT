import streamlit as st
import fitz  # PyMuPDF
import os
from groq import Groq
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ======= 🔐 Load Secrets from .env =======
load_dotenv()

def ask_groq(prompt):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content

# ======= 📄 PDF to Text =======
def extract_text(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ======= 🔍 Chunking =======
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.create_documents([text])

# ======= 🧠 Embeddings + FAISS =======
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# ======= 🧠 RAG =======
def generate_answer(vectorstore, question):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful AI assistant. Use the context below to answer the user's question accurately.

Context:
{context}

Question:
{question}

Answer only using the provided context. If the answer is not available in the provided context, say:
"The answer is not available in the provided context."
    """.strip()

    print("🔎 Retrieved Context:\n", context)
    print("❓ User Question:", question)

    return ask_groq(prompt)

# ======= 🚀 Streamlit App =======
st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📚 Chat With Your PDF (Groq + HuggingFace + FAISS)")

pdf = st.file_uploader("Upload your PDF file here", type="pdf")

if pdf:
    # Use file name and size as a unique key for caching
    pdf_key = f"{pdf.name}_{pdf.size}"
    
    if "pdf_key" not in st.session_state or st.session_state.pdf_key != pdf_key:
        with st.status("Processing PDF...", expanded=True) as status:
            status.write("Extracting text...")
            text = extract_text(pdf)
            
            status.write("Splitting into chunks...")
            docs = get_chunks(text)
            
            status.write("Generating embeddings...")
            vectorstore = create_vector_store(docs)
            
            # Save to session state so it doesn't re-run on user inputs
            st.session_state.docs = docs
            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_key = pdf_key
            status.update(label="✅ PDF processed and ready!", state="complete", expanded=False)
    else:
        # Show a static success indicator if already processed
        st.success("✅ PDF processed and ready (loaded from cache)!")
        
    docs = st.session_state.docs
    vectorstore = st.session_state.vectorstore
    st.markdown("---")

    if st.checkbox("🔍 Show extracted chunks (debug)"):
        for i, doc in enumerate(docs[:5]):
            st.markdown(f"**Chunk {i+1}:**\n```\n{doc.page_content[:500]}\n```")

    question = st.text_input("Ask a question based on the PDF:")
    if question:
        with st.spinner("🔎 Generating answer..."):
            try:
                answer = generate_answer(vectorstore, question)
                st.markdown("### 📥 Answer:")
                st.success(answer)
            except Exception as e:
                st.error("Something went wrong.")
                st.code(str(e))

