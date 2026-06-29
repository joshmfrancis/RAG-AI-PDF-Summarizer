# Retrieval-Augmented Generation (RAG) — Local Document Q&A

A full-stack web application that lets you upload a document and ask natural language questions about it. All inference runs locally — no external API calls, no data leaves your machine.

---

## What is RAG?

Retrieval-Augmented Generation is a technique for grounding a language model's responses in a specific set of documents. Instead of relying on the model's training data (which may be outdated, incomplete, or unrelated to your content), RAG retrieves the most relevant passages from your document and feeds them directly into the model's context window before generating a response.

This solves two key problems with vanilla LLMs:

- **Hallucination** — the model can only answer from what you gave it, so it cannot invent facts from outside the document.
- **Knowledge scope** — the model does not need to have been trained on your document. You can ask questions about any private or domain-specific content.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Docker Compose                        │
│                                                             │
│  ┌─────────────────────┐      ┌──────────────────────────┐  │
│  │    FastAPI Backend  │      │         Ollama            │  │
│  │    (port 8000)      │◄────►│  tinyllama  (port 11434) │  │
│  │                     │      │  nomic-embed-text         │  │
│  │  LangChain          │      │                          │  │
│  │  ChromaDB           │      │  Local CPU inference     │  │
│  │  FastAPI            │      │  No external API calls   │  │
│  └─────────────────────┘      └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## RAG Pipeline

### Ingestion (upload phase)

1. **Document loading** — PDF, TXT, or MD files are read using LangChain's `PyPDFLoader` or `TextLoader`.
2. **Chunking** — The document is split into 800-token chunks with 100-token overlap using `RecursiveCharacterTextSplitter`. The overlap prevents answers from being cut off at chunk boundaries.
3. **Embedding** — Each chunk is converted into a vector embedding using `nomic-embed-text`, a lightweight embedding model running locally via Ollama.
4. **Storage** — Embeddings are stored in a ChromaDB vector database persisted to disk inside the container.

### Retrieval + Generation (query phase)

1. **Query embedding** — The user's question is embedded using the same `nomic-embed-text` model.
2. **Similarity search** — ChromaDB performs a cosine similarity search and returns the top 4 most relevant chunks.
3. **Prompt construction** — The retrieved chunks are concatenated into a context block and injected into a prompt template along with the user's question.
4. **Generation** — The prompt is sent to `tinyllama` via Ollama's `/api/generate` endpoint. The response streams back token by token via Server-Sent Events.
5. **Display** — The frontend reads the SSE stream and renders each token as it arrives.

---

## Why Two Models?

| Model              | Role                               | Size    |
| ------------------ | ---------------------------------- | ------- |
| `nomic-embed-text` | Converts text to vector embeddings | ~274 MB |
| `tinyllama`        | Generates natural language answers | ~637 MB |

These are separate concerns. Embedding models produce fixed-length vectors optimized for semantic similarity — they are not designed to generate text. Generation models take a prompt and produce a continuation. Using a dedicated embedding model produces significantly better retrieval quality than using the generation model for both tasks.

---

## Chunking Strategy

```
chunk_size    = 800 tokens   # balance between context richness and retrieval precision
chunk_overlap = 100 tokens   # prevents answers from being split at boundaries
k             = 4 chunks     # top-k retrieved per query (~3,200 tokens of context)
separators    = ["\n\n", "\n", ".", " "]  # prefer natural break points
```

The chunk size was chosen to fit comfortably within tinyllama's context window while providing enough context per chunk for meaningful retrieval. Larger chunks would improve context but reduce retrieval precision; smaller chunks improve precision but may truncate answers.

---

## Tech Stack

| Layer            | Technology                                 |
| ---------------- | ------------------------------------------ |
| LLM              | TinyLlama 1.1B (Q4_0 quantized) via Ollama |
| Embeddings       | nomic-embed-text v1.5 via Ollama           |
| Vector store     | ChromaDB                                   |
| Orchestration    | LangChain                                  |
| Backend          | FastAPI + Python 3.11                      |
| Frontend         | Vanilla HTML/JS (no framework)             |
| Containerization | Docker Compose                             |

---

## Setup

### Prerequisites

- Docker Desktop (with at least 4GB RAM allocated)
- ~2GB free disk space for model weights

### Run

```bash
# Clone (SSH)
git clone git@github.com:joshmfrancis/RAG-AI-PDF-Summarizer.git
cd RAG-AI-PDF-Summarizer

# OR clone via HTTPS
# git clone https://github.com/joshmfrancis/RAG-AI-PDF-Summarizer.git
# cd RAG-AI-PDF-Summarizer

docker compose up --build
```

On first run, the `ollama-pull` container will download `tinyllama` (~637MB) and `nomic-embed-text` (~274MB). This is a one-time download. Watch for:

```
rag-ollama-pull exited with code 0
```

That means both models are ready. Then open **http://localhost:8000**.

### Using a different model

Change `OLLAMA_MODEL` in `docker-compose.yml` and update the pull command accordingly. Any model on [ollama.com/library](https://ollama.com/library) works. Larger models (llama3, mistral) produce better answers but are significantly slower on CPU.

---

## API Endpoints

| Method | Path      | Description                                |
| ------ | --------- | ------------------------------------------ |
| `GET`  | `/`       | Web UI                                     |
| `GET`  | `/status` | Returns model info and document load state |
| `GET`  | `/health` | Checks Ollama connectivity                 |
| `POST` | `/upload` | Upload a document (PDF, TXT, MD)           |
| `POST` | `/ask`    | Ask a question — returns SSE stream        |

---

## Project Structure

```
rag-app/
├── docker-compose.yml
└── app/
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py              # FastAPI backend + RAG pipeline
    └── static/
        └── index.html       # Frontend (single file, no framework)
```
