import streamlit as st
from dotenv import load_dotenv

# --- Our helper functions ---
from helpers.chunker import chunk_data
from helpers.youtubeloader import load_from_youtube
from helpers.vectorstore import create_vector_store
from helpers.retriever import create_retriever
from helpers.chain import create_rag_chain


# --- Page Setup ---
st.set_page_config(page_title="YouTube Q&A", layout="wide")
st.title("Ask Questions to any YouTube Video 💬")

# Load environment variables (for API keys like GROQ_API_KEY)
load_dotenv()

# --- Initialize session state ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Sidebar: YouTube Input ---
with st.sidebar:
    st.header("Setup")
    youtube_url = st.text_input("Enter YouTube URL:")

    if st.button("Process Video"):
        if youtube_url:
            with st.spinner("Processing video... This may take a few minutes."):
                try:
                    # ✅ Use our helper to get transcript
                    transcript_docs = load_from_youtube(youtube_url)

                    # ✅ Split transcript into chunks
                    chunks = chunk_data(transcript_docs)

                    # ✅ Create embeddings + vector store
                    vector_store = create_vector_store(chunks)

                    # ✅ Build retriever
                    st.session_state.retriever = create_retriever(vector_store)

                    # ✅ Create RAG chain (retriever + LLM)
                    st.session_state.rag_chain = create_rag_chain(
                        st.session_state.retriever
                    )

                    st.success("Video processed successfully!")

                except Exception as e:
                    st.error(f"Error occurred: {e}")
        else:
            st.warning("Please enter a YouTube URL.")

# --- Main Section: Q&A ---
st.header("Q&A")
if st.session_state.rag_chain:
    st.info("Ready to answer questions.")
    question = st.text_input("Ask a question about the video:")

    if question:
        with st.spinner("Generating answer..."):
            # Ask question through the RAG chain
            answer = st.session_state.rag_chain.invoke(question)
            st.write(answer)
else:
    st.info("Please enter a YouTube URL in the sidebar and click 'Process Video'.")
