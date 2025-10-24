# Minimal test version - helps isolate the crash
import streamlit as st
import os

st.set_page_config(page_title="GoodBlue RAG - Minimal Test", layout="wide")

st.title("🔍 Minimal Test Version")
st.write("This version tests basic functionality step by step.")

# Test 1: API Key
st.header("Test 1: API Key Loading")
try:
    key = st.secrets.get("OPENAI_API_KEY", "")
    if key:
        st.success(f"✅ API Key found: {key[:15]}...")
    else:
        st.error("❌ No API key found")
except Exception as e:
    st.error(f"❌ Error: {e}")

# Test 2: OpenAI Client
st.header("Test 2: OpenAI Client")
try:
    from openai import OpenAI
    client = OpenAI(api_key=key)
    st.success("✅ Client initialized")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# Test 3: Simple Embedding
st.header("Test 3: Simple Embedding Test")
if st.button("Test Embedding API"):
    try:
        with st.spinner("Testing..."):
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=["Hello world"]
            )
            vec = resp.data[0].embedding
            st.success(f"✅ Got {len(vec)}-dimensional vector!")
            st.write(f"First 5 values: {vec[:5]}")
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Test 4: File Upload (no processing)
st.header("Test 4: File Upload Test")
uploaded = st.file_uploader("Upload a test file", type=["txt", "pdf"])
if uploaded:
    st.success(f"✅ File uploaded: {uploaded.name}")
    st.write(f"Size: {uploaded.size} bytes")
    st.write(f"Type: {uploaded.type}")

st.divider()
st.caption("If all tests pass, the issue is in the main app logic. If any fail, that's the problem.")
