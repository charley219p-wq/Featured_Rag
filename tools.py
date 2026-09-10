# Merged tools.py
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embeddings import GenerativeEngineEmbeddings

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
embeddings = GenerativeEngineEmbeddings(
        api_key=os.getenv("GENAI_API_KEY"),
        model="amazon.titan-embed-text-v2:0"
)

_vectorstore = None

@tool
def web_search(query:str)->str:
    """Search the web."""
    results=tavily.search(query=query,max_results=3)
    out=[]
    for r in results["results"]:
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}")
    return "\n-----\n".join(out)

@tool
def scrape_url(url:str)->str:
    """Scrape a URL."""
    try:
        r=requests.get(url,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
        soup=BeautifulSoup(r.text,"html.parser")
        for t in soup(["script","style","nav","footer"]):
            t.decompose()
        return soup.get_text(separator=" ",strip=True)[:3000]
    except Exception as e:
        return str(e)

def load_pdf_to_vectorstore(pdf_path:str,chunk_size=1000,chunk_overlap=150):
    global _vectorstore
    docs=PyPDFLoader(pdf_path).load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks=splitter.split_documents(docs)
    _vectorstore=FAISS.from_documents(chunks,embeddings)
    return len(chunks)

@tool
def pdf_search(query:str)->str:
    """Search loaded PDF."""
    global _vectorstore
    if _vectorstore is None:
        return "No PDF loaded."
    docs=_vectorstore.similarity_search(query,k=5)
    return "\n\n-----\n\n".join(
        f"[Page {d.metadata.get('page','?')}]\n{d.page_content}" for d in docs
    )

def get_full_pdf_summary_context(max_chunks=15):
    global _vectorstore
    if _vectorstore is None:
        return ""
    docs=_vectorstore.similarity_search("summary overview key findings",k=max_chunks)
    return "\n\n-----\n\n".join(
        f"[Page {d.metadata.get('page','?')}]\n{d.page_content}" for d in docs
    )