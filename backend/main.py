from fastapi import FastAPI, HTTPException
from ytscrapper import scrapper
from vector_store import create_vector_store
from embeddings import get_embeddings
from session_store import (
    SESSION_CHAT_HISTORY,
    SESSION_VECTOR_STORES
)
from chat_model import get_chat_model, chat_model_invoke
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import uuid

load_dotenv()

app = FastAPI(
    title="YouTube Scrapper API",
    servers=[
        {
            "url": "https://youtube-scrapper-2.onrender.com",
            "description": "Production"
        }
    ]
)

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "YouTube Scrapper RAG API"
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------
# SCRAPE ENDPOINT (SAFE)
# ------------------------------------
@app.get("/scrape/{repo_name}")
def create_session_vector_store(repo: str):
    session_id = str(uuid.uuid4())

    try:
        docs = scrapper(repo)
    except Exception as e:
        print("Scraping failed:", e)
        raise HTTPException(
            status_code=503,
            detail="API not working right now"
        )

    embeddings = get_embeddings()

    vector_store = create_vector_store(
        embeddings=embeddings,
        documents=docs
    )

    SESSION_VECTOR_STORES[session_id] = vector_store
    SESSION_CHAT_HISTORY[session_id] = []

    return {
        "session_id": session_id,
        "message": "Scraping successful"
    }

# ------------------------------------
# CHAT ENDPOINT
# ------------------------------------
@app.post("/chat")
def chat(session_id: str, query: str):
    if session_id not in SESSION_VECTOR_STORES:
        raise HTTPException(
            status_code=404,
            detail="Session expired or invalid"
        )

    vector_store = SESSION_VECTOR_STORES[session_id]
    chat_model = get_chat_model()

    chat_history = SESSION_CHAT_HISTORY.get(session_id, [])
    chat_history.append({"role": "user", "content": query})

    retriever = vector_store.as_retriever(k=15)
    answer = chat_model_invoke(chat_model, chat_history, retriever)

    chat_history.append({"role": "assistant", "content": answer})
    SESSION_CHAT_HISTORY[session_id] = chat_history

    return {"answer": answer}
