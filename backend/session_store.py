from typing import Dict
from langchain_community.vectorstores import FAISS
from typing import List


# Vector store per session
SESSION_VECTOR_STORES: Dict[str, FAISS] = {}
# Chat history per session
SESSION_CHAT_HISTORY: Dict[str, List[dict]] = {}

# Last activity timestamps (optional but recommended)
SESSION_LAST_ACTIVE: Dict[str, float] = {}