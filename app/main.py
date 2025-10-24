# Streamlit entry point; holds layout/sidebar and privacy toggle.
import streamlit as st
from io import StringIO
from pypdf import PdfReader

st.title("GoodBlue — RAG Starter")

tab_upload, tab_manage, tab_ask = st.tabs(["📤 Upload", "🗂️ Manage corpus", "❓ Ask questions"])

with tab_upload:
    st.subheader("Upload your source files")
    st.caption("PDF or TXT. Multiple files supported.")
    files = st.file_uploader("Drag & drop or browse", type=["pdf", "txt"], accept_multiple_files=True, key="uploader_main")

    if files:
        texts = []
        for f in files:
            if f.type == "application/pdf":
                reader = PdfReader(f)
                texts.append("\n".join([(p.extract_text() or "") for p in reader.pages]))
            else:
                texts.append(StringIO(f.getvalue().decode("utf-8", errors="ignore")).read())
        st.success(f"Loaded {len(files)} file(s).")
        st.session_state["raw_text"] = "\n\n".join(texts)

with tab_manage:
    st.subheader("Manage corpus")
    if "raw_text" in st.session_state:
        st.write("First 800 chars of your corpus:")
        st.code(st.session_state["raw_text"][:800])
    else:
        st.info("No corpus yet. Go to **Upload** first.")

with tab_ask:
    st.subheader("Ask questions")
    if "raw_text" not in st.session_state:
        st.warning("Please upload files first in the **Upload** tab.")
    else:
        q = st.text_input("Your question")
        if st.button("Test"):
            st.write("(Stub) You asked:", q)

