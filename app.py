import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
from docx import Document as DocxDocument

load_dotenv()

APP_TITLE = "Chat With Your Documents"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
PERSIST_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "document_knowledge"

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )

def extract_text(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if suffix == ".docx":
        doc = DocxDocument(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Supported formats: PDF, TXT, DOCX.")

def make_documents(uploaded_files):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks, ids = [], []
    for uploaded in uploaded_files:
        text = extract_text(uploaded)
        if not text.strip():
            st.warning(f"No readable text found in {uploaded.name}. Scanned PDFs may need OCR.")
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        source_docs = [Document(
            page_content=text,
            metadata={"source": uploaded.name, "file_id": digest}
        )]
        split_docs = splitter.split_documents(source_docs)
        for index, chunk in enumerate(split_docs):
            chunk.metadata["chunk"] = index
            chunks.append(chunk)
            ids.append(f"{digest}-{index}")
    return chunks, ids

def format_context(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'Unknown')} | chunk {d.metadata.get('chunk', '?')}]\n{d.page_content}"
        for d in docs
    )

st.title("📚 RAG: Chat With Your Documents")
st.caption("Upload documents → chunk text → create embeddings → store vectors → retrieve relevant context → ask Ollama.")

with st.sidebar:
    st.header("Model settings")
    st.write(f"**Chat model:** `{CHAT_MODEL}`")
    st.write(f"**Embedding model:** `{EMBED_MODEL}`")
    st.write(f"**Ollama:** `{OLLAMA_BASE_URL}`")
    st.divider()
    st.info("Start Ollama and download both models before using this app.")
    st.code("ollama pull llama3.2\nollama pull nomic-embed-text", language="bash")
    if st.button("Clear knowledge base", type="secondary"):
        try:
            get_vector_store().delete_collection()
            get_vector_store.clear()
            st.success("Knowledge base cleared. Refresh the page.")
        except Exception as exc:
            st.error(f"Could not clear database: {exc}")

tab_upload, tab_chat = st.tabs(["1. Upload & index", "2. Ask questions"])

with tab_upload:
    st.subheader("Add your documents")
    uploads = st.file_uploader(
        "Choose PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    if st.button("Process documents", type="primary", disabled=not uploads):
        try:
            with st.spinner("Reading, chunking, embedding, and saving documents..."):
                chunks, ids = make_documents(uploads)
                if not chunks:
                    st.warning("No text chunks were created.")
                else:
                    store = get_vector_store()
                    store.add_documents(documents=chunks, ids=ids)
                    st.success(f"Indexed {len(uploads)} file(s) into {len(chunks)} text chunks.")
                    st.session_state["indexed"] = True
        except Exception as exc:
            st.error("Document processing failed. Check that Ollama is running and the embedding model is installed.")
            st.exception(exc)

with tab_chat:
    st.subheader("Ask questions about your files")
    question = st.text_input("Your question", placeholder="What are the main points in these documents?")
    top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=8, value=4)
    if st.button("Get answer", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Searching documents and generating an answer..."):
                store = get_vector_store()
                matches = store.similarity_search(question, k=top_k)
                if not matches:
                    st.warning("No indexed content found. Upload and process documents first.")
                else:
                    context = format_context(matches)
                    prompt = f"""You are a helpful document question-answering assistant.
Answer using only the context below. If the answer is not present, clearly say you cannot find it in the uploaded documents.
Do not invent facts. Keep the answer clear and structured.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
                    llm = ChatOllama(
                        model=CHAT_MODEL,
                        base_url=OLLAMA_BASE_URL,
                        temperature=0.2
                    )
                    response = llm.invoke(prompt)
                    st.markdown("### Answer")
                    st.write(response.content)
                    with st.expander("Retrieved context / sources"):
                        for i, doc in enumerate(matches, 1):
                            st.markdown(f"**{i}. {doc.metadata.get('source', 'Unknown')} — chunk {doc.metadata.get('chunk', '?')}**")
                            st.write(doc.page_content[:1200] + ("..." if len(doc.page_content) > 1200 else ""))
        except Exception as exc:
            st.error("Could not generate an answer. Confirm Ollama is running and both models are installed.")
            st.exception(exc)

st.divider()
st.caption("Built with Streamlit, LangChain, Ollama, and ChromaDB. Documents stay in your local project folder.")
