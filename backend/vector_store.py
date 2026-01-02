from langchain_core.vectorstores import InMemoryVectorStore
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever

def create_vector_store(embeddings, documents):
    vector_store = InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    return vector_store

def add_documents_to_vector_store(vector_store, embeddings, documents):
    vector_store.add_documents(
        documents=documents,
        embedding=embeddings,
    )
    return vector_store

def create_retriever_from_vector_store(vector_store, k):
    
    retriever = GraphRetriever(
    store = vector_store,
    edges = [("path", "path"), ("source", "source")],
    strategy = Eager(k=k, start_k=1, max_depth=2),
    )
    return retriever
def retrieve_documents(retriever, query):
    results = retriever.retrieve(query)
    return results



