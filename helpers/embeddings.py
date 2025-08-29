from langchain_community.embeddings import HuggingFaceEmbeddings

def load_embeddings():
    """
    Load Hugging Face sentence-transformer embeddings safely for Streamlit Cloud.
    Force CPU to avoid 'meta tensor' errors when GPU is not available.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  # Force CPU
        )
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None
