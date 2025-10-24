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
import faiss  # from faiss-cpu

# --------------------------- App Setup --------------------------- #
st.set_page_config(page_title="GoodBlue — RAG Starter", layout="wide")
load_dotenv()

def get_client():
    """
    Lazy-init OpenAI client. Checks Streamlit secrets first, then .env
    Returns None if no key; callers should guard.
    """
    # Try Streamlit secrets first (for cloud deployment)
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except (AttributeError, FileNotFoundError):
        # Fall back to environment variable (for local development)
        key = os.getenv("OPENAI_API_KEY", "")
    
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key)

# --------------------------- Helpers ----------------------------- #
def read_pdf(file) -> str:
    """Extract text from a PDF file-like object."""
    try:
        reader = PdfReader(file)
        texts = []
        for p in reader.pages:
            texts.append(p.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        st.error(f"PDF reading error: {str(e)}")
        raise

def read_txt(file) -> str:
    """Read text from a TXT file-like object."""
    try:
        return StringIO(file.getvalue().decode("utf-8", errors="ignore")).read()
    except Exception as e:
        st.error(f"TXT reading error: {str(e)}")
        raise

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
        st.error("❌ OPENAI_API_KEY missing. Check Streamlit secrets or .env file.")
        st.stop()
    
    try:
        # Log what we're sending
        st.write(f"  → Sending {len(texts)} texts to OpenAI ({sum(len(t) for t in texts)} total chars)")
        
        resp = client.embeddings.create(model=model, input=texts)
        
        st.write(f"  ✓ Received {len(resp.data)} embeddings")
        vecs = [d.embedding for d in resp.data]
        return np.array(vecs, dtype="float32")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        st.error(f"❌ OpenAI API Error ({error_type}): {error_msg}")
        st.write("**Full error details:**")
        st.code(traceback.format_exc())
        
        # Specific error handling
        if "rate_limit" in error_msg.lower():
            st.warning("🕐 Rate limit hit. Try again in a few minutes or reduce batch size.")
        elif "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
            st.warning("💳 Quota exceeded. Check your OpenAI billing at https://platform.openai.com/account/billing")
        elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            st.warning("🔑 Authentication failed. Verify your API key is correct.")
        elif "model" in error_msg.lower():
            st.warning(f"🤖 Model '{model}' not accessible. Check your API tier/permissions.")
        
        raise

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Create a cosine-similarity FAISS index by normalizing and using inner product."""
    try:
        faiss.normalize_L2(vectors)
        idx = faiss.IndexFlatIP(vectors.shape[1])
        idx.add(vectors)
        return idx
    except Exception as e:
        st.error(f"❌ FAISS indexing error: {str(e)}")
        raise

def search(idx: faiss.IndexFlatIP, query_vec: np.ndarray, k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Search top-k; returns (distances, indices)."""
    try:
        q = query_vec.copy()
        faiss.normalize_L2(q)
        D, I = idx.search(q, k)
        return D, I
    except Exception as e:
        st.error(f"❌ Search error: {str(e)}")
        raise

def llm_answer(query: str, context_chunks: List[str]) -> str:
    """LLM answer grounded strictly in provided context with simple [C#] citations."""
    client = get_client()
    if client is None:
        st.error("❌ OPENAI_API_KEY missing. Check Streamlit secrets or .env file.")
        st.stop()
    
    try:
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
    except Exception as e:
        st.error(f"❌ OpenAI API Error: {str(e)}")
        raise

# ------------------------ Session State -------------------------- #
defaults = {
    "raw_texts": [],
    "chunks": [],
    "embeddings": None,
    "index": None,
    "last_uploaded_files": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------- UI -------------------------------- #
st.title("GoodBlue — RAG Starter")

# Debug section
with st.expander("🔍 Debug Info (click to expand)"):
    client = get_client()
    if client:
        st.success("✅ OpenAI client initialized successfully")
    else:
        st.error("❌ Failed to initialize OpenAI client")
    
    st.write("**Checking API key source:**")
    try:
        has_secret = "OPENAI_API_KEY" in st.secrets
        st.write(f"- Streamlit secret exists: {has_secret}")
        if has_secret:
            key = st.secrets.get("OPENAI_API_KEY", "")
            st.write(f"- Key starts with: `{key[:10]}...`")
            st.write(f"- Key length: {len(key)} characters")
    except Exception as e:
        st.write(f"- Streamlit secrets error: {e}")
    
    env_key = os.getenv("OPENAI_API_KEY", "")
    st.write(f"- Environment variable exists: {bool(env_key)}")
    if env_key:
        st.write(f"- Env key starts with: `{env_key[:10]}...`")
    
    st.write(f"**Session state:**")
    st.write(f"- Chunks loaded: {len(st.session_state.chunks)}")
    st.write(f"- Index exists: {st.session_state.index is not None}")
    
    st.divider()
    st.write("**Test OpenAI API:**")
    if st.button("🧪 Test Embedding API", key="test_api"):
        with st.spinner("Testing..."):
            try:
                test_vec = embed_texts(["Hello world test"])
                st.success(f"✅ API works! Got {len(test_vec[0])} dimensional vector.")
            except Exception as e:
                st.error(f"❌ API test failed: {str(e)}")

tab_upload, tab_manage, tab_ask = st.tabs(["📤 Upload", "🗂️ Manage corpus", "❓ Ask questions"])

# --------------------------- Upload Tab -------------------------- #
with tab_upload:
    st.subheader("Upload your source files")
    st.caption("PDF or TXT. Multiple files supported.")
    files = st.file_uploader(
        "Drag & drop or browse",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="uploader_main",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        chunk_size = st.number_input("Chunk size", min_value=300, max_value=2000, value=900, step=50)
    with c2:
        overlap = st.number_input("Overlap", min_value=0, max_value=400, value=150, step=10)
    with c3:
        top_k_default = st.number_input("Default Top-K", min_value=1, max_value=10, value=4, step=1)

    if files:
        # Prevent re-processing on every rerun
        file_names = [f.name for f in files]
        if st.session_state.last_uploaded_files != file_names:
            st.session_state.last_uploaded_files = file_names
            
            texts = []
            with st.status("Reading files...", expanded=True) as status:
                for f in files:
                    try:
                        st.write(f"📄 Reading {f.name}...")
                        if f.type == "application/pdf":
                            texts.append(read_pdf(f))
                        else:
                            texts.append(read_txt(f))
                        st.write(f"✅ {f.name}")
                    except Exception as e:
                        st.error(f"❌ Error reading {f.name}: {str(e)}")
                        st.code(traceback.format_exc())
                        continue
                status.update(label="Files read!", state="complete")
            
            st.session_state.raw_texts = [t for t in texts if t]

            if st.session_state.raw_texts:
                full_text = "\n\n".join(st.session_state.raw_texts)
                st.session_state.chunks = simple_chunk(full_text, chunk_size=int(chunk_size), overlap=int(overlap))

                if st.session_state.chunks:
                    num_chunks = len(st.session_state.chunks)
                    st.info(f"📄 Extracted {num_chunks} chunks from your files.")
                    
                    # Warn if too many chunks
                    if num_chunks > 500:
                        st.warning(f"⚠️ Large corpus ({num_chunks} chunks). This will take several minutes.")
                    
                    with st.status("Embedding & indexing...", expanded=True) as status:
                        try:
                            # Batch process - smaller batches for stability
                            batch_size = 50
                            all_vecs = []
                            
                            progress_bar = st.progress(0)
                            
                            for i in range(0, num_chunks, batch_size):
                                batch = st.session_state.chunks[i:i+batch_size]
                                batch_num = (i // batch_size) + 1
                                total_batches = (num_chunks + batch_size - 1) // batch_size
                                
                                st.write(f"🔄 Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                                progress_bar.progress(i / num_chunks)
                                
                                batch_vecs = embed_texts(batch)
                                all_vecs.append(batch_vecs)
                                
                                # Delay to respect rate limits
                                if i + batch_size < num_chunks:
                                    time.sleep(1)
                            
                            progress_bar.progress(1.0)
                            st.write("🔨 Building search index...")
                            
                            # Combine all vectors
                            vecs = np.vstack(all_vecs)
                            st.session_state.embeddings = vecs
                            st.session_state.index = build_faiss_index(vecs)
                            
                            status.update(label=f"✅ Indexed {num_chunks} chunks!", state="complete")
                        except Exception as e:
                            st.error(f"❌ Failed: {str(e)}")
                            st.code(traceback.format_exc())
                            status.update(label="❌ Failed", state="error")
                else:
                    st.warning("No text content detected in your files.")
            else:
                st.warning("Could not extract text from uploaded files.")
        else:
            # Files already processed
            if st.session_state.chunks:
                st.success(f"✅ {len(st.session_state.chunks)} chunks already loaded.")
                st.caption("Upload new files to replace current corpus.")

# -------------------------- Manage Tab --------------------------- #
with tab_manage:
    st.subheader("Corpus status")
    if not st.session_state.chunks:
        st.info("No corpus loaded yet. Upload in the **Upload** tab.")
    else:
        st.write(f"**Chunks:** {len(st.session_state.chunks)}")
        st.write("**Preview (first 600 chars of the first chunk):**")
        st.code(st.session_state.chunks[0][:600] + ("…" if len(st.session_state.chunks[0]) > 600 else ""))

        with st.expander("Show 5 random chunks"):
            import random
            for i, idx in enumerate(random.sample(range(len(st.session_state.chunks)), min(5, len(st.session_state.chunks))), start=1):
                st.markdown(f"**Chunk #{idx}**")
                snippet = st.session_state.chunks[idx]
                st.caption(snippet[:500] + ("…" if len(snippet) > 500 else ""))

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear corpus"):
                st.session_state.raw_texts = []
                st.session_state.chunks = []
                st.session_state.embeddings = None
                st.session_state.index = None
                st.session_state.last_uploaded_files = None
                st.success("Cleared.")
                st.rerun()
        with c2:
            if st.button("Re-embed & re-index"):
                if st.session_state.chunks:
                    with st.spinner("Rebuilding index…"):
                        try:
                            vecs = embed_texts(st.session_state.chunks)
                            st.session_state.embeddings = vecs
                            st.session_state.index = build_faiss_index(vecs)
                            st.success("Index rebuilt.")
                        except Exception as e:
                            st.error(f"Failed to rebuild index: {str(e)}")
                else:
                    st.warning("No chunks to re-index.")

# --------------------------- Ask Tab ----------------------------- #
with tab_ask:
    st.subheader("Ask questions")

    # Ensure we have a corpus
    if not st.session_state.chunks:
        st.warning("⚠️ Please upload files first in the **Upload** tab.")
        st.stop()

    # Ensure embeddings & index exist
    if st.session_state.index is None or st.session_state.embeddings is None:
        st.warning("⚠️ No search index found.")
        if st.button("🔨 Embed & index now", type="secondary"):
            with st.spinner("Embedding & indexing…"):
                try:
                    vecs = embed_texts(st.session_state.chunks)
                    st.session_state.embeddings = vecs
                    st.session_state.index = build_faiss_index(vecs)
                    st.success(f"Indexed {len(st.session_state.chunks)} chunks.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to embed/index: {str(e)}")
        st.stop()

    # Query UI
    q = st.text_input("Your question", placeholder="What would you like to know?")
    top_k = st.slider("Top-K retrieval", 1, 10, value=4, step=1)
    go = st.button("🔍 Search & Answer", type="primary", disabled=not q)

    if go and q:
        try:
            with st.spinner("Retrieving relevant chunks…"):
                q_vec = embed_texts([q])
                D, I = search(st.session_state.index, q_vec, k=top_k)
                selected = [st.session_state.chunks[i] for i in I[0] if 0 <= i < len(st.session_state.chunks)]

            with st.spinner("Generating answer…"):
                answer = llm_answer(q, selected)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### Answer")
                st.write(answer)
            with col2:
                st.markdown("### Sources")
                for rank, idx in enumerate(I[0], start=1):
                    if 0 <= idx < len(st.session_state.chunks):
                        st.markdown(f"**C{rank}** (score: {float(D[0][rank-1]):.3f})")
                        chunk = st.session_state.chunks[idx]
                        with st.expander(f"View chunk {rank}"):
                            st.caption(chunk)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.code(traceback.format_exc())

# ------------------------- Footer -------------------------------- #
st.divider()
st.caption("GoodBlue RAG Starter • Powered by OpenAI + FAISS")
