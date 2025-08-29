from langchain_community.embeddings import HuggingFaceEmbeddings

def load_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  # ✅ force CPU
        )
        return embeddings
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        return None

