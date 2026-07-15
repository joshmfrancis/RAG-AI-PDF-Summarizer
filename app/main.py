import os
import shutil
import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

UPLOAD_DIR = Path("/tmp/uploads")
CHROMA_DIR = Path("/tmp/chroma")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

app = FastAPI(title="RAG Q&A")
app.mount("/static", StaticFiles(directory="/app/static"), name="static")

vectorstore = None


class QuestionRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=Path("/app/static/index.html").read_text())


@app.get("/status")
async def status():
    return {
        "document_loaded": vectorstore is not None,
        "model": OLLAMA_MODEL,
        "embed_model": OLLAMA_EMBED_MODEL,
        "ollama_url": OLLAMA_BASE_URL,
    }


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            return {"ollama": "ok", "available_models": models}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {e}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore

    # Check embed model is ready
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": "test"},
                timeout=10
            )
            if r.status_code != 200:
                raise HTTPException(status_code=503, detail="Embed model not ready. Wait a moment and try again.")
    except httpx.TimeoutException:
        raise HTTPException(status_code=503, detail="Ollama is not responding.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a PDF, TXT, or MD file.")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(str(dest)) if suffix == ".pdf" else TextLoader(str(dest))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise HTTPException(status_code=400, detail="No content could be extracted from the file.")

    embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)

    # Reset vectorstore — use in-memory Chroma to avoid SQLite locking issues
    vectorstore = None

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return {
        "filename": file.filename,
        "pages": len(docs),
        "chunks": len(chunks),
        "message": f"Ingested {len(docs)} page(s) → {len(chunks)} chunks. Ready for questions."
    }


@app.post("/ask")
async def ask_question(body: QuestionRequest):
    global vectorstore

    if vectorstore is None:
        raise HTTPException(status_code=400, detail="No document loaded. Upload a file first.")
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Retrieve relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = await asyncio.get_event_loop().run_in_executor(
        None, lambda: retriever.get_relevant_documents(body.question)
    )
    context = "\n\n".join(d.page_content for d in docs)

    prompt_template = """
You are a strict assistant that ONLY answers questions based on the provided context. 
If the answer to the question cannot be explicitly found in the context below, you 
must respond exactly with: "I am sorry, but the provided document does not contain that information." 
Do not use your own outside knowledge under any circumstances.

Context:
{context}

Question: {question}
Answer:"""

    prompt = prompt_template.format(context=context, question=body.question)

    # Serialize chunk metadata to send to frontend
    chunk_data = json.dumps([
        {
            "content": d.page_content,
            "metadata": d.metadata
        }
        for d in docs
    ])

    async def stream_tokens() -> AsyncGenerator[str, None]:
        # Send chunk info first
        yield f"data: [CHUNKS]{chunk_data}\n\n"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield f"data: {token}\n\n"
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")
