import os
from dotenv import load_dotenv
from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
def scrapper(repo_name:str):

    TEXT_EXTENSIONS = (
        ".py", ".js", ".ts", ".java", ".go",
        ".md", ".txt", ".rst",
        ".json", ".yaml", ".yml",
        ".html", ".css",
        ".sh", ".env",
    )
    loader = GithubFileLoader(
        repo="ayushhhh2999/RAG_ChatBot_using_weaviate",

        access_token=os.getenv("GITHUB_TOKEN"),
        branch="main",
        file_filter=lambda p: p.lower().endswith(TEXT_EXTENSIONS),

    )

    docs = loader.load()
    #print(len(docs))
    #print(docs[0].page_content[:500])


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_documents(docs)
    print(len(texts))
    print(texts[500].page_content)
    return texts

