import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def process_document(file_path):
    """
    Loads a PDF or TXT file, splits it into chunks,
    generates embeddings, and saves the FAISS index locally.
    """
    # Step 1: Load document depending on file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or TXT file.")

    pages = loader.load_and_split()

    # Step 2: Split content into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)

    # Step 3: Create vector store using sentence-transformers
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # Step 4: Save the vector index for later use
    doc_name = Path(file_path).stem
    save_path = os.path.join("data", f"{doc_name}_index")
    vector_store.save_local(save_path)

    return vector_store
