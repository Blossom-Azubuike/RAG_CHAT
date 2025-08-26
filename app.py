import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from helpers.chain import create_rag_chain
from helpers.chunker import chunk_data
from helpers.pdfloader import load_pdf
from helpers.youtubeloader import load_from_youtube
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Zubi- Chat", layout="wide")
st.title("Zubi - Let's Chat 💬")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- PDF Upload ---
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
if pdf_file:
    try:
        text = load_pdf(pdf_file)
        chunks = chunk_data([Document(page_content=text)])
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.session_state.retriever = vector_store.as_retriever()
        st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
        st.session_state.chat_history = []
        st.success("PDF processed successfully!")

    except Exception as e:
        st.error(f"Error processing PDF: {e}")

# --- YouTube URL ---
youtube_url = st.text_input("Or enter a YouTube URL:")
if youtube_url:
    try:
        yt_docs = load_from_youtube(youtube_url)
        chunks = chunk_data(yt_docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.session_state.retriever = vector_store.as_retriever()
        st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
        st.session_state.chat_history = []
        st.success("YouTube transcript processed successfully!")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Manual Memory Reset ---
if st.button("Reset Memory"):
    st.session_state.chat_history = []
    st.success("Memory cleared!")

# --- Q&A Section ---
question = st.text_input("Ask a question:")
if question:
    try:
        if st.session_state.rag_chain:
            answer = st.session_state.rag_chain.invoke(question)
        else:
            # General chat if no PDF or YouTube
            llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
            from langchain_core.prompts import ChatPromptTemplate
            formatted_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                ("user", "{question}")
            ]).format_prompt(question=question).to_messages()
            answer_obj = llm.invoke(formatted_prompt)
            answer = answer_obj.content if hasattr(answer_obj, "content") else str(answer_obj)

        st.session_state.chat_history.append({"question": question, "answer": answer})

    except Exception as e:
        st.error(f"Error: {e}")

# --- Display Chat History ---
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Bot:** {chat['answer']}")
    st.markdown("---")

st.markdown("Developed by Azubuike Blossom (https://blossom-azubuike.github.io/Me/Project.html)")
