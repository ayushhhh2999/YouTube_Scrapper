from fastapi import FastAPI
from ytscrapper import scrapper
from vector_store import create_vector_store, add_documents_to_vector_store, create_retriever_from_vector_store, retrieve_documents
from embeddings import get_embeddings
import os
from session_store import (
    SESSION_CHAT_HISTORY,
    SESSION_LAST_ACTIVE, 
    SESSION_VECTOR_STORES
    )
import uuid
from fastapi import HTTPException
from chat_model import get_chat_model, chat_model_invoke
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

@app.get("/scrape/{repo_name}")
def create_session_vector_store(repo: str) -> dict:
    session_id = str(uuid.uuid4())

    docs = scrapper(repo)
    embeddings = get_embeddings()

    vector_store = create_vector_store(
        embeddings=embeddings,
           # IMPORTANT
        documents=docs   
    )

    SESSION_VECTOR_STORES[session_id] = vector_store
    SESSION_CHAT_HISTORY[session_id] = []
    return {"session_id": session_id}

@app.post("/chat")
def chat(session_id: str, query: str):
    if session_id not in SESSION_VECTOR_STORES:
        raise HTTPException(
            status_code=404,
            detail="Session expired or invalid"
        )
    chat_model = get_chat_model()
    vector_store = SESSION_VECTOR_STORES.get(session_id)

    if not vector_store:
        return {"error": "Session expired or invalid"}
    if session_id not in SESSION_CHAT_HISTORY:
        SESSION_CHAT_HISTORY[session_id] = []

    chat_history = SESSION_CHAT_HISTORY[session_id]
    chat_history.append({"role": "user", "content": query})

    retriever = vector_store.as_retriever(k=15)
    answer = chat_model_invoke(chat_model, chat_history, retriever)
    chat_history.append({"role": "assistant", "content": answer})
    return {"answer": answer}
