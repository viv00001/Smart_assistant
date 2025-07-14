import os
import streamlit as st
from utils import process_document
from rag_backend import quick_summary, answer_query
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")  

llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4o", temperature=0.2)

st.title("Smart Assistant for Research Summarization")

uploaded_file = st.file_uploader("Upload a research document", type=["pdf", "txt"])

if uploaded_file:
    st.success("File uploaded successfully!")
    
    # Save and process
    save_path = f"data/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Indexing and summarizing..."):
        store = process_document(save_path)
        chunks = store.similarity_search(" ", k=10)  # Get sample chunks
        summary = quick_summary(chunks, llm)

    st.subheader("Document Summary (≈150 words)")
    st.write(summary)

    st.subheader("Ask Anything About the Document")
    user_question = st.text_input("Enter your question here:")

    if user_question:
        with st.spinner("Generating answer..."):
            response = answer_query(user_question, store, llm)
        st.markdown("**Answer:**")
        st.write(response)
