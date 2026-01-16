from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uuid
import os
import shutil

from typing import List

from ytscrapper import scrapper
from embeddings import get_embeddings
from vector_store import create_vector_store
from session_store import SESSION_CHAT_HISTORY, SESSION_VECTOR_STORES
from chat_model import get_chat_model, chat_model_invoke

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

app = FastAPI(
    title="Unified RAG API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------------
# UTIL FUNCTIONS
# ----------------------------------

def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".csv":
        return CSVLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path).load()
    else:
        return UnstructuredFileLoader(file_path).load()


def chunk_documents(docs, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)

# ----------------------------------
# HEALTH CHECK
# ----------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "Unified RAG API"}

# ----------------------------------
# FEATURE 1: REPO SCRAPER (UNCHANGED)
# ----------------------------------
@app.get("/scrape/repo")
def scrape_repo(repo: str):
    session_id = str(uuid.uuid4())

    try:
        docs = scrapper(repo)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Repo scraping failed")

    embeddings = get_embeddings()
    vector_store = create_vector_store(
        embeddings=embeddings,
        documents=docs
    )

    SESSION_VECTOR_STORES[session_id] = vector_store
    SESSION_CHAT_HISTORY[session_id] = []

    return {
        "session_id": session_id,
        "message": "Repository scraped successfully"
    }

# ----------------------------------
# FEATURE 2: DOCUMENT INGESTION
# ----------------------------------
@app.post("/scrape/document")
async def upload_document(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        docs = load_document(file_path)
        chunks = chunk_documents(docs)

        embeddings = get_embeddings()
        vector_store = create_vector_store(
            embeddings=embeddings,
            documents=chunks
        )

        SESSION_VECTOR_STORES[session_id] = vector_store
        SESSION_CHAT_HISTORY[session_id] = []

        return {
            "session_id": session_id,
            "message": "Document processed successfully",
            "chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# ----------------------------------
# CHAT (WORKS FOR BOTH)
# ----------------------------------
@app.post("/chat")
def chat(session_id: str, query: str):
    if session_id not in SESSION_VECTOR_STORES:
        raise HTTPException(status_code=404, detail="Invalid or expired session")

    vector_store = SESSION_VECTOR_STORES[session_id]
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    chat_history = SESSION_CHAT_HISTORY.get(session_id, [])
    chat_history.append({"role": "user", "content": query})

    chat_model = get_chat_model()
    answer = chat_model_invoke(chat_model, chat_history, retriever)

    chat_history.append({"role": "assistant", "content": answer})
    SESSION_CHAT_HISTORY[session_id] = chat_history

    return {"answer": answer}
