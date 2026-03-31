from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uuid
import os
import shutil

from typing import List
import time
from threading import Thread, Event
from time import sleep
from ytscrapper import scrapper
from embeddings import get_embeddings
from vector_store import create_vector_store
from session_store import SESSION_CHAT_HISTORY, SESSION_VECTOR_STORES, SESSION_LAST_ACTIVE
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

SESSION_TTL_SECONDS = 24 * 60 * 60
MAX_CHAT_HISTORY_MESSAGES = 10
CLEANUP_INTERVAL_SECONDS = 60  # run cleanup every minute

cleanup_stop_event = Event()


def cleanup_expired_sessions():
    now = time.time()
    expired = [sid for sid, ts in SESSION_LAST_ACTIVE.items() if now - ts > SESSION_TTL_SECONDS]
    for sid in expired:
        SESSION_LAST_ACTIVE.pop(sid, None)
        SESSION_VECTOR_STORES.pop(sid, None)
        SESSION_CHAT_HISTORY.pop(sid, None)


def cleanup_worker():
    while not cleanup_stop_event.is_set():
        try:
            cleanup_expired_sessions()
        except Exception:
            pass
        cleanup_stop_event.wait(CLEANUP_INTERVAL_SECONDS)


def prune_chat_history(chat_history):
    # Keep only the most recent MAX_CHAT_HISTORY_MESSAGES records
    if len(chat_history) > MAX_CHAT_HISTORY_MESSAGES:
        return chat_history[-MAX_CHAT_HISTORY_MESSAGES:]
    return chat_history

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
    cleanup_expired_sessions()
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
    SESSION_LAST_ACTIVE[session_id] = time.time()
    return {
        "session_id": session_id,
        "message": "Repository scraped successfully"
    }

# ----------------------------------
# FEATURE 2: DOCUMENT INGESTION
# ----------------------------------
@app.post("/scrape/document")
async def upload_document(file: UploadFile = File(...)):
    cleanup_expired_sessions()
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
        SESSION_LAST_ACTIVE[session_id] = time.time()
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
@app.on_event("startup")
def on_startup():
    cleanup_thread = Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()


@app.on_event("shutdown")
def on_shutdown():
    cleanup_stop_event.set()


@app.post("/chat")
def chat(session_id: str, query: str):
    cleanup_expired_sessions()
    if session_id not in SESSION_VECTOR_STORES:
        raise HTTPException(status_code=404, detail="Invalid or expired session")

    vector_store = SESSION_VECTOR_STORES[session_id]
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    chat_history = SESSION_CHAT_HISTORY.get(session_id, [])
    chat_history.append({"role": "user", "content": query})
    chat_history = prune_chat_history(chat_history)

    chat_model = get_chat_model()
    answer = chat_model_invoke(chat_model, chat_history, retriever)

    chat_history.append({"role": "assistant", "content": answer})
    chat_history = prune_chat_history(chat_history)

    SESSION_CHAT_HISTORY[session_id] = chat_history
    SESSION_LAST_ACTIVE[session_id] = time.time()
    return {"answer": answer}
@app.get("/sessions")
def list_sessions():
    return {
        "active_sessions": list(SESSION_VECTOR_STORES.keys())
    }