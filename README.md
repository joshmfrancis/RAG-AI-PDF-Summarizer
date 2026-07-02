# Retrieval-Augmented Generation (RAG) — Local Document Q&A

A full-stack web application that lets you upload a document and ask natural language questions about it. All inference runs locally — no external API calls, no data leaves your machine.

**Repo:** https://github.com/joshmfrancis/RAG-AI-PDF-Summarizer

---

## What is RAG?

Retrieval-Augmented Generation is a technique for grounding a language model's responses in a specific set of documents. Instead of relying on the model's training data (which may be outdated, incomplete, or unrelated to your content), RAG retrieves the most relevant passages from your document and feeds them directly into the model's context window before generating a response.

This solves two key problems with vanilla LLMs:

- **Hallucination** — the model can only answer from what you gave it, so it cannot invent facts from outside the document.
- **Knowledge scope** — the model does not need to have been trained on your document. You can ask questions about any private or domain-specific content.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Docker Compose                           │
│                                                                  │
│  ┌──────────────────────┐       ┌──────────────────────────────┐ │
│  │   FastAPI Backend    │       │          Ollama              │ │
│  │   (port 8000)        │◄─────►│  tinyllama   (port 11434)    │ │
│  │                      │       │  nomic-embed-text            │ │
│  │  LangChain           │       │                              │ │
│  │  ChromaDB (memory)   │       │  Local CPU inference         │ │
│  │  Vanilla HTML/JS     │       │  No external API calls       │ │
│  └──────────────────────┘       └──────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────┐                                        │
│  │   ollama-pull        │  One-shot container that downloads     │
│  │   (init container)   │  tinyllama + nomic-embed-text          │
│  └──────────────────────┘                                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## RAG Pipeline

### Ingestion (upload phase)

1. **Document loading** — PDF, TXT, or MD files are read using LangChain's `PyPDFLoader` or `TextLoader`.
2. **Chunking** — The document is split into 800-token chunks with 100-token overlap using `RecursiveCharacterTextSplitter`. The overlap prevents answers from being cut off at chunk boundaries.
3. **Embedding** — Each chunk is converted into a high-dimensional vector using `nomic-embed-text`, a dedicated embedding model running locally via Ollama.
4. **Storage** — Embeddings are stored in ChromaDB running fully in-memory. No files are written to disk — the vectorstore is rebuilt fresh on each upload and lives only for the duration of the session.

### Retrieval + Generation (query phase)

1. **Query embedding** — The user's question is embedded using the same `nomic-embed-text` model, producing a vector in the same space as the document chunks.
2. **Similarity search** — ChromaDB performs a cosine similarity search and returns the top 4 most semantically relevant chunks.
3. **Prompt construction** — The retrieved chunks are joined into a context block and injected into a prompt template along with the user's question.
4. **Generation** — The prompt is sent directly to `tinyllama` via Ollama's `/api/generate` HTTP endpoint. The response streams back token by token using Server-Sent Events (SSE).
5. **Display** — The frontend reads the SSE stream and renders each token as it arrives, with a blinking cursor while generating. After the answer, the source chunks used to generate it are shown in a collapsible section.

---

## Why Two Models?

| Model              | Role                               | Size    |
| ------------------ | ---------------------------------- | ------- |
| `nomic-embed-text` | Converts text to vector embeddings | ~274 MB |
| `tinyllama`        | Generates natural language answers | ~637 MB |

These are separate concerns. Embedding models produce fixed-length vectors optimized for semantic similarity search — they are not designed to generate text. Generation models take a prompt and produce a continuation. Using a dedicated embedding model produces significantly better retrieval quality than asking the generation model to do both.

---

## Chunking Strategy

```
chunk_size    = 800 tokens     # balances context richness vs. retrieval precision
chunk_overlap = 100 tokens     # prevents answers from being cut off at boundaries
k             = 4 chunks       # top-k retrieved per query
separators    = ["\n\n", "\n", ".", " "]   # prefer natural break points
```

Chunk size was chosen to fit within TinyLlama's 2048-token context window while leaving room for the prompt template and generated response. Larger chunks give more context per result but reduce retrieval precision; smaller chunks improve precision but may split answers mid-sentence.

---

## Why In-Memory ChromaDB?

ChromaDB supports two modes: persistent (SQLite on disk) and in-memory. This project uses in-memory for simplicity and reliability. The persistent mode requires careful lifecycle management — if the database connection isn't properly closed before deleting the directory on re-upload, SQLite holds a file lock and the next write fails. In-memory avoids this entirely: the vectorstore is a Python object that gets replaced on each upload with no file system involvement.

The tradeoff is that embeddings are lost if the container restarts, but for a single-session document Q&A tool this is acceptable behavior.

---

## Streaming Architecture

The `/ask` endpoint uses FastAPI's `StreamingResponse` with `text/event-stream` (SSE). The backend:

1. Retrieves chunks synchronously from ChromaDB
2. Sends a `[CHUNKS]` SSE event with the chunk metadata as JSON
3. Opens an async HTTP stream to Ollama's `/api/generate`
4. Forwards each token as a `data:` SSE event as it arrives
5. Sends `[DONE]` when generation is complete

The frontend reads this stream with the `ReadableStream` API and renders tokens incrementally.

---

## Tech Stack

| Layer              | Technology                                  |
| ------------------ | ------------------------------------------- |
| LLM                | TinyLlama 1.1B (Q4_0 quantized) via Ollama  |
| Embeddings         | nomic-embed-text v1.5 via Ollama            |
| Vector store       | ChromaDB (in-memory)                        |
| Orchestration      | LangChain                                   |
| Backend            | FastAPI + Python 3.11                       |
| Frontend           | Vanilla HTML/JS — no framework, single file |
| Containerization   | Docker Compose (3 services)                 |
| Markdown rendering | marked.js (CDN)                             |

---

## Setup

### Prerequisites

- Docker Desktop with at least 4GB RAM allocated
- ~2GB free disk space for model weights

### Run

```bash
git clone https://github.com/joshmfrancis/RAG-AI-PDF-Summarizer.git
cd RAG-AI-PDF-Summarizer

docker compose up --build
```

On first run, the `ollama-pull` container downloads `tinyllama` (~637MB) and `nomic-embed-text` (~274MB). This is a one-time download. Watch for:

```
rag-ollama-pull exited with code 0
```

That means both models are ready. Open **http://localhost:8000**, upload a PDF, and start asking questions.

### Using a different model

Change `OLLAMA_MODEL` in `docker-compose.yml` and update the pull command in the `ollama-pull` service. Any model on [ollama.com/library](https://ollama.com/library) works. Larger models (llama3, mistral) produce better answers but are significantly slower on CPU without a GPU.

---

## API Endpoints

| Method | Path      | Description                                                         |
| ------ | --------- | ------------------------------------------------------------------- |
| `GET`  | `/`       | Web UI                                                              |
| `GET`  | `/readme` | Returns raw README markdown                                         |
| `GET`  | `/status` | Returns model info and document load state                          |
| `GET`  | `/health` | Checks Ollama connectivity and lists available models               |
| `POST` | `/upload` | Upload a document (PDF, TXT, MD) — chunks, embeds, stores in memory |
| `POST` | `/ask`    | Ask a question — returns SSE stream of tokens + chunk metadata      |

---

## Project Structure

```
RAG-AI-PDF-Summarizer/
├── docker-compose.yml       # 3 services: ollama, ollama-pull, app
└── app/
    ├── Dockerfile
    ├── requirements.txt
    ├── README.md            # served at /readme endpoint
    ├── main.py              # FastAPI backend + RAG pipeline
    └── static/
        └── index.html       # Frontend — dark mode, collapsible chunks, README modal
```
