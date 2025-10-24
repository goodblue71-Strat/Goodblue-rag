# Streamlit entry point; holds layout/sidebar and privacy toggle.
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
