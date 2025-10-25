# C12
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

def simple_chunk(text: str, chunk_size: int = 900, overlap: int = 150, progress_callback=None) -> List[str]:
    """Naive character-based chunking with overlap."""
    if not text:
        return []
    
    chunks = []
    n, start = len(text), 0
    max_chunks = 5000  # Safety limit
    
    while start < n:
        if len(chunks) >= max_chunks:
            raise ValueError(f"Too many chunks (>{max_chunks}). Your file is too large. Try: 1) Increase chunk_size to 1500-2000, or 2) Split your document into smaller files.")
        
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if len(chunk) > 10:
            chunks.append(chunk)
        
        # Report progress every 100 chunks
        if progress_callback and len(chunks) % 100 == 0:
            progress_callback(len(chunks), n, start)
        
        # Move forward by (chunk_size - overlap)
        next_start = end - overlap
        # Safety: if we're not moving forward, jump to end instead
        if next_start <= start:
            next_start = end
        start = next_start
    
    return chunks

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Return a 2D numpy array (N, d) of embedding vectors."""
    client = get_client()
    if client is None:
        raise ValueError("OPENAI_API_KEY missing")
    
    # Retry logic for transient failures
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=model, 
                input=texts,
                timeout=30.0  # 30 second timeout
            )
            vecs = [d.embedding for d in resp.data]
            return np.array(vecs, dtype="float32")
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                st.warning(f"⚠️ Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Last attempt failed
                error_msg = str(e)
                if "timeout" in error_msg.lower():
                    raise Exception(f"Connection timeout. Try reducing batch size or check your internet connection.")
                elif "rate_limit" in error_msg.lower():
                    raise Exception(f"Rate limit hit. Wait a minute and try again.")
                elif "quota" in error_msg.lower():
                    raise Exception(f"OpenAI quota exceeded. Check billing at https://platform.openai.com/account/billing")
                else:
                    raise Exception(f"OpenAI API error: {error_msg}")

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
        chunk_size = st.number_input(
            "Chunk size", 
            300, 3000, 1500,  # Changed default from 900 to 1500
            50,
            help="Larger chunks = fewer total chunks = faster processing. Recommended: 1500-2000 for large files."
        )
    with col2:
        overlap = st.number_input(
            "Overlap", 
            0, 400, 150, 10,
            help="How many characters overlap between chunks"
        )
    
    # Process button
    if files and st.button("📊 Process Files", type="primary", disabled=st.session_state.processing):
        # Generate unique ID for this upload
        file_id = "-".join([f.name for f in files]) + str(time.time())
        
        # Only process if different files
        if file_id != st.session_state.processed_file_id:
            st.session_state.processing = True
            
            try:
                # Step 1: Read files
                st.write("🔵 STEP 1: Reading files...")
                with st.status("Reading files...") as status:
                    texts = []
                    for f in files:
                        st.write(f"📄 {f.name}")
                        if f.type == "application/pdf":
                            texts.append(read_pdf(f))
                        else:
                            texts.append(read_txt(f))
                    status.update(label="✅ Files read", state="complete")
                
                st.write(f"✅ STEP 1 COMPLETE: Read {len(texts)} files")
                
                # Step 2: Join text
                st.write("🔵 STEP 2: Joining text...")
                full_text = "\n\n".join([t for t in texts if t])
                text_length = len(full_text)
                st.write(f"✅ STEP 2 COMPLETE: {text_length:,} characters")
                
                # Check if file is too large
                max_chars = 1_000_000  # 1 million characters (~500 pages)
                if text_length > max_chars:
                    st.error(f"❌ File too large: {text_length:,} characters")
                    st.error(f"Maximum supported: {max_chars:,} characters")
                    st.info("💡 Try splitting your document into smaller files")
                    st.session_state.processing = False
                    st.stop()
                
                # Step 3: Chunk
                st.write("🔵 STEP 3: Chunking text...")
                st.write(f"  → Chunk size: {chunk_size}, Overlap: {overlap}")
                
                chunk_progress = st.empty()
                
                def chunk_callback(num_chunks, total_chars, current_pos):
                    pct = int(100 * current_pos / total_chars)
                    chunk_progress.write(f"  → {num_chunks} chunks created ({pct}% processed)")
                
                try:
                    chunks = simple_chunk(full_text, chunk_size, overlap, progress_callback=chunk_callback)
                    chunk_progress.empty()
                    st.write(f"✅ STEP 3 COMPLETE: Created {len(chunks)} chunks")
                    
                    # Warn if very large
                    if len(chunks) > 1000:
                        st.warning(f"⚠️ Large corpus: {len(chunks)} chunks will take ~{len(chunks) * 2 // 60} minutes to embed")
                        st.info("💡 Tip: Increase 'Chunk size' to 1500-2000 to reduce processing time")
                        
                except Exception as e:
                    st.error(f"❌ Chunking failed: {str(e)}")
                    st.session_state.processing = False
                    raise
                
                if len(chunks) == 0:
                    st.error("No text extracted from files")
                    st.session_state.processing = False
                    st.stop()
                
                st.info(f"📄 Ready to embed {len(chunks)} chunks")
                
                if len(chunks) > 200:
                    st.warning(f"⚠️ Large file: {len(chunks)} chunks. This may take several minutes.")
                
                # Step 4: Embed in small batches to avoid timeout
                st.write("🔵 STEP 4: Starting embedding...")
                with st.status("Embedding...", expanded=True) as status:
                    batch_size = 20  # Much smaller to avoid timeout
                    all_vecs = []
                    progress = st.progress(0)
                    
                    total_batches = (len(chunks) + batch_size - 1) // batch_size
                    st.write(f"Will process {total_batches} batches of ~{batch_size} chunks each")
                    
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i+batch_size]
                        batch_num = i // batch_size + 1
                        
                        st.write(f"🔄 Batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                        
                        try:
                            vecs = embed_texts(batch)
                            all_vecs.append(vecs)
                            st.write(f"✓ Batch {batch_num} complete")
                        except Exception as e:
                            st.error(f"❌ Failed on batch {batch_num}: {str(e)}")
                            st.session_state.processing = False
                            raise
                        
                        progress.progress(min(1.0, (i + batch_size) / len(chunks)))
                        
                        # Longer delay between batches
                        if i + batch_size < len(chunks):
                            time.sleep(1.5)
                    
                    status.update(label=f"✅ Embedded {len(chunks)} chunks", state="complete")
                
                st.write(f"✅ STEP 4 COMPLETE: Embedded all chunks")
                
                # Step 5: Build index
                st.write("🔵 STEP 5: Building search index...")
                with st.spinner("Building index..."):
                    combined_vecs = np.vstack(all_vecs)
                    idx = build_faiss_index(combined_vecs)
                st.write(f"✅ STEP 5 COMPLETE: Index built")
                
                # Update state
                st.session_state.chunks = chunks
                st.session_state.embeddings = combined_vecs
                st.session_state.index = idx
                st.session_state.processed_file_id = file_id
                st.session_state.processing = False
                
                st.success(f"✅ Indexed {len(chunks)} chunks!")
                
            except Exception as e:
                st.error(f"❌ CRASH DETECTED!")
                st.error(f"Error: {str(e)}")
                st.error(f"Type: {type(e).__name__}")
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
