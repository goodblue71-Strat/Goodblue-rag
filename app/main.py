# app/main.py
# GoodBlue — RAG Starter (Upload → Chunk → Embed (FAISS) → Retrieve → Answer with Sources)

import os
from io import StringIO
from typing import List, Tuple
import time
import traceback

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import faiss

# --------------------------- App Setup --------------------------- #
st.set_page_config(page_title="GoodBlue — RAG Starter", layout="wide")
load_dotenv()

def get_client():
    """Lazy-init OpenAI client."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except (AttributeError, FileNotFoundError):
        key = os.getenv("OPENAI_API_KEY", "")
    
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key)

# --------------------------- Helpers ----------------------------- #
def read_pdf(file) -> str:
    """Extract text from a PDF file-like object."""
    reader = PdfReader(file)
    texts = []
    for p in reader.pages:
        texts.append(p.extract_text() or "")
    return "\n".join(texts)

def read_txt(file) -> str:
    """Read text from a TXT file-like object."""
    return StringIO(file.getvalue().decode("utf-8", errors="ignore")).read()

def simple_chunk(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Naive character-based chunking with overlap."""
    if not text:
        return []
    chunks = []
    n, start = len(text), 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if len(chunk) > 10:
            chunks.append(chunk)
        start = max(0, end - overlap)
    return chunks

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Return a 2D numpy array (N, d) of embedding vectors."""
    client = get_client()
    if client is None:
        raise ValueError("OPENAI_API_KEY missing")
    
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Create a cosine-similarity FAISS index."""
    faiss.normalize_L2(vectors)
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    return idx

def search(idx: faiss.IndexFlatIP, query_vec: np.ndarray, k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Search top-k; returns (distances, indices)."""
    q = query_vec.copy()
    faiss.normalize_L2(q)
    D, I = idx.search(q, k)
    return D, I

def llm_answer(query: str, context_chunks: List[str]) -> str:
    """LLM answer with citations."""
    client = get_client()
    if client is None:
        raise ValueError("OPENAI_API_KEY missing")
    
    system_msg = (
        "You are a helpful strategy analyst. Answer ONLY using the provided context. "
        "Cite chunk numbers like [C1], [C2]. If the answer isn't in context, say you don't know."
    )
    numbered = "\n\n".join([f"[C{i+1}] {c}" for i, c in enumerate(context_chunks)])
    user_msg = f"Question: {query}\n\nContext:\n{numbered}"
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ------------------------ Session State -------------------------- #
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "index" not in st.session_state:
    st.session_state.index = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None

# ----------------------------- UI -------------------------------- #
st.title("GoodBlue — RAG Starter")

with st.expander("🔍 Debug Info"):
    client = get_client()
    st.write(f"✅ API Key: {'Loaded' if client else 'Missing'}")
    st.write(f"📦 Chunks: {len(st.session_state.chunks)}")
    st.write(f"🔍 Index: {'Ready' if st.session_state.index else 'Not ready'}")

tab_upload, tab_ask = st.tabs(["📤 Upload", "❓ Ask"])

# --------------------------- Upload Tab -------------------------- #
with tab_upload:
    st.subheader("Upload your documents")
    
    files = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk size", 300, 2000, 900, 50)
    with col2:
        overlap = st.number_input("Overlap", 0, 400, 150, 10)
    
    # Process button
    if files and st.button("📊 Process Files", type="primary", disabled=st.session_state.processing):
        # Generate unique ID for this upload
        file_id = "-".join([f.name for f in files]) + str(time.time())
        
        # Only process if different files
        if file_id != st.session_state.processed_file_id:
            st.session_state.processing = True
            
            try:
                # Step 1: Read files
                with st.status("Reading files...") as status:
                    texts = []
                    for f in files:
                        st.write(f"📄 {f.name}")
                        if f.type == "application/pdf":
                            texts.append(read_pdf(f))
                        else:
                            texts.append(read_txt(f))
                    status.update(label="✅ Files read", state="complete")
                
                # Step 2: Chunk
                full_text = "\n\n".join([t for t in texts if t])
                chunks = simple_chunk(full_text, chunk_size, overlap)
                st.info(f"📄 Created {len(chunks)} chunks")
                
                # Step 3: Embed in batches
                with st.status("Embedding...") as status:
                    batch_size = 50
                    all_vecs = []
                    progress = st.progress(0)
                    
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i+batch_size]
                        st.write(f"Batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                        
                        vecs = embed_texts(batch)
                        all_vecs.append(vecs)
                        
                        progress.progress(min(1.0, (i + batch_size) / len(chunks)))
                        if i + batch_size < len(chunks):
                            time.sleep(0.5)
                    
                    status.update(label="✅ Embedded", state="complete")
                
                # Step 4: Build index
                with st.spinner("Building index..."):
                    combined_vecs = np.vstack(all_vecs)
                    idx = build_faiss_index(combined_vecs)
                
                # Update state
                st.session_state.chunks = chunks
                st.session_state.embeddings = combined_vecs
                st.session_state.index = idx
                st.session_state.processed_file_id = file_id
                st.session_state.processing = False
                
                st.success(f"✅ Indexed {len(chunks)} chunks!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.code(traceback.format_exc())
                st.session_state.processing = False
        else:
            st.info("✅ These files are already processed")
    
    # Show status
    if st.session_state.chunks:
        st.success(f"📊 Corpus ready: {len(st.session_state.chunks)} chunks indexed")
        
        if st.button("🗑️ Clear corpus"):
            st.session_state.chunks = []
            st.session_state.embeddings = None
            st.session_state.index = None
            st.session_state.processed_file_id = None
            st.rerun()

# --------------------------- Ask Tab ----------------------------- #
with tab_ask:
    st.subheader("Ask questions")
    
    if not st.session_state.chunks:
        st.warning("⚠️ Please upload and process files first")
        st.stop()
    
    if not st.session_state.index:
        st.warning("⚠️ No search index found")
        st.stop()
    
    query = st.text_input("Your question", placeholder="What would you like to know?")
    top_k = st.slider("Results to retrieve", 1, 10, 4)
    
    if st.button("🔍 Search & Answer", type="primary", disabled=not query):
        try:
            with st.spinner("Searching..."):
                q_vec = embed_texts([query])
                D, I = search(st.session_state.index, q_vec, k=top_k)
                selected = [st.session_state.chunks[i] for i in I[0] if 0 <= i < len(st.session_state.chunks)]
            
            with st.spinner("Generating answer..."):
                answer = llm_answer(query, selected)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Answer")
                st.write(answer)
            
            with col2:
                st.markdown("### Sources")
                for rank, idx in enumerate(I[0], start=1):
                    if 0 <= idx < len(st.session_state.chunks):
                        score = float(D[0][rank-1])
                        st.markdown(f"**C{rank}** (score: {score:.3f})")
                        with st.expander("View"):
                            st.caption(st.session_state.chunks[idx][:500])
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.code(traceback.format_exc())

st.divider()
st.caption("Powered by OpenAI + FAISS")
