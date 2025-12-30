from fastapi import FastAPI, UploadFile
import shutil

from ingest import extract_text
from chunking import chunk_pages
from vectorstore import store_chunks
from rag import generate_answer

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile):
    path = f"data/uploads/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pages = extract_text(path)
    chunks = chunk_pages(pages)
    store_chunks(chunks)

    return {
        "status": "uploaded",
        "chunks_stored": len(chunks)
    }

@app.post("/ask")
async def ask(question: str):
    return generate_answer(question)


from fastapi import FastAPI
from gemini_client import generate_answer

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Legal RAG API is running 🚀",
        "endpoints": ["/ask"]
    }

@app.get("/ask")
def ask(q: str):
    return {"answer": generate_answer(q)}

