import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource(show_spinner="Loading embeddings model...")
def load_embeddings():
    """
    Load and cache HuggingFace sentence transformer embeddings.
    """
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"⚠️ Failed to load embeddings: {e}")
        return None
