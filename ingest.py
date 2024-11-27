import os
import torch
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Limit PyTorch threads
torch.set_num_threads(1)

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/faiss_db'  # Path without extension for LangChain

# Create vector database with metadata
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': device})

    # Create the vector store
    vector_store = FAISS.from_documents(texts, embeddings)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

    # Save the vector store (creates `.faiss` and `.pkl` files)
    vector_store.save_local(DB_FAISS_PATH)
    
    print(f"FAISS vector store created and saved at {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
