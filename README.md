# RAG System — Web & PDF Knowledge Retrieval

> A Retrieval-Augmented Generation (RAG) pipeline built and run entirely in **Google Colab**. Ingests content from **live web pages** and **PDF documents**, stores semantic embeddings in a managed vector database, and answers natural-language questions using a large language model grounded entirely in retrieved context.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How RAG Works](#how-rag-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Configuration](#configuration)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [1. Document Ingestion](#1-document-ingestion)
  - [2. Text Chunking](#2-text-chunking)
  - [3. Embedding Generation](#3-embedding-generation)
  - [4. Vector Storage](#4-vector-storage)
  - [5. Retrieval & Generation](#5-retrieval--generation)
- [Usage](#usage)
- [Example Queries](#example-queries)
- [Design Decisions](#design-decisions)
- [Limitations & Known Issues](#limitations--known-issues)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

Large Language Models (LLMs) are powerful but have a fundamental limitation: their knowledge is frozen at training time. They cannot access private documents, recent web content, or domain-specific data without retraining.

This project solves that problem using **RAG (Retrieval-Augmented Generation)** — a technique that dynamically injects relevant, up-to-date context into every LLM prompt at query time. Instead of relying on the model's parametric memory, the system fetches the most semantically relevant chunks of text from a vector database and passes them to the model as grounding context.

This implementation supports two document sources:

- **Web pages** — scraped and parsed at runtime via URL
- **PDF files** — loaded page-by-page from local disk or mounted storage

Both sources are unified into a single retrieval pipeline backed by **Qdrant** (a high-performance vector database) and powered by **Cohere** embeddings and chat models.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                        │
│                                                              │
│   Web URLs ──► WebBaseLoader ──┐                             │
│                                ├──► Combined Document List   │
│   PDF Files ──► PyPDFLoader ───┘                             │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                       TEXT CHUNKING                          │
│                                                              │
│   RecursiveCharacterTextSplitter                             │
│   chunk_size=1000  |  chunk_overlap=200                      │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    EMBEDDING GENERATION                      │
│                                                              │
│   Cohere embed-english-v3.0                                  │
│   Dense vectors (1024 dimensions)                            │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     VECTOR STORAGE                           │
│                                                              │
│   Qdrant Cloud  ──  Collection: RAG_web-pdf                  │
│   Cosine similarity index                                    │
└──────────────────────────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │     QUERY TIME        │
              │                       │
         User Query              Top-K Chunks
              │                       │
              ▼                       ▼
        Embed Query ──► Similarity Search ──► Context Assembly
                                                     │
                                                     ▼
                                          Cohere command-r-08-2024
                                                     │
                                                     ▼
                                              Final Answer
```

---

## How RAG Works

RAG is a two-phase process: **indexing** (done once) and **retrieval + generation** (done at query time).

### Phase 1 — Indexing

1. **Load** — Raw content is loaded from URLs and/or PDF files.
2. **Split** — Long documents are broken into smaller, overlapping text chunks. Overlap preserves sentence context that would otherwise be lost at chunk boundaries.
3. **Embed** — Each chunk is converted to a high-dimensional numerical vector (embedding) that captures its semantic meaning. Similar text produces similar vectors.
4. **Store** — Embeddings and their associated text are persisted in a vector database (Qdrant), enabling fast approximate nearest-neighbour search.

### Phase 2 — Retrieval and Generation

1. **Query embedding** — The user's question is embedded using the same model as the documents.
2. **Similarity search** — The query vector is compared against all stored document vectors. The top-K most similar chunks are retrieved.
3. **Prompt construction** — Retrieved chunks are concatenated into a context block and injected into a structured prompt alongside the user's question.
4. **LLM generation** — The LLM reads only the provided context (not its training data) to produce a grounded, factual answer.

This approach eliminates hallucination on domain-specific topics because the model is explicitly instructed to answer from the supplied context only.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Notebook Environment | Google Colab | Development & execution — no local setup required |
| Web Loader | LangChain `WebBaseLoader` | Scrape and parse web pages |
| PDF Loader | LangChain `PyPDFLoader` | Extract text from PDF files |
| Text Splitter | LangChain `RecursiveCharacterTextSplitter` | Chunk documents into manageable pieces |
| Embedding Model | Cohere `embed-english-v3.0` | Convert text to dense semantic vectors |
| Vector Database | Qdrant Cloud | Store and search vector embeddings |
| Chat / LLM | Cohere `command-r-08-2024` | Generate answers from retrieved context |
| LangChain | `langchain`, `langchain-cohere`, `langchain-qdrant` | Orchestration layer |
| Secret Management | Colab Secrets (`userdata`) | Store API keys safely, outside notebook code |

---

## Project Structure

```
RAG_Web_PDF.ipynb          # Main notebook — full pipeline
│
├── Install Libraries       # pip install block
├── Import Libraries        # All imports
├── Config                  # Models, collection name, chunking params
├── Load Document           # WebBaseLoader + PyPDFLoader
├── Create Vector           # Embeddings + Qdrant collection setup
└── Building RAG System     # queryRag() function + example queries
```

---

## Getting Started

This project is designed to run entirely in **Google Colab**. No local Python installation or environment setup is required.

### Prerequisites

- A Google account (to access Colab)
- A [Cohere](https://cohere.com) account — free tier is sufficient
- A [Qdrant Cloud](https://cloud.qdrant.io) account — free tier cluster is sufficient
- Your PDF file uploaded to Colab's `/content/` directory

### Open in Colab

Upload `RAG_Web_PDF.ipynb` to [Google Colab](https://colab.research.google.com) by going to **File → Upload notebook**, or open it directly from your Google Drive if you store it there.

### Installation

The first cell installs all required libraries into the Colab runtime. Just run it:

```python
!pip install -qU langchain langchain-text-splitters langchain-cohere langchain-qdrant langchain-community
!pip install -qU qdrant-client beautifulsoup4 lxml
!pip install -qU pypdf
```

> These installs are session-scoped. They will need to re-run each time the Colab runtime restarts.

### Setting Up Secrets in Colab

This project uses **Colab's built-in Secrets manager** to store API keys — no `.env` files, no hardcoded credentials.

1. In your Colab notebook, click the **🔑 key icon** in the left sidebar (Secrets).
2. Add the following three secrets:

| Secret Name | Where to Get It |
|---|---|
| `COHERE_API_KEY` | [Cohere dashboard](https://dashboard.cohere.com/api-keys) |
| `QDRANT_API_KEY` | Qdrant Cloud → your cluster → API Keys |
| `QDRANT_URL` | Qdrant Cloud → your cluster → Cluster URL (e.g. `https://xyz.europe-west3-0.gcp.cloud.qdrant.io`) |

3. Enable **"Notebook access"** for each secret using the toggle next to it.

The Config cell then loads them automatically:

```python
from google.colab import userdata

os.environ['COHERE_API_KEY'] = userdata.get('COHERE_API_KEY')
os.environ['QDRANT_API_KEY'] = userdata.get('QDRANT_API_KEY')
os.environ['QDRANT_URL']     = userdata.get('QDRANT_URL')
```

> **Never paste API keys directly into notebook cells.** Colab Secrets keep them out of your code and out of version history when you push to GitHub.

### Uploading Your PDF

PDF files must be uploaded to the Colab session storage before running the notebook:

1. In the left sidebar, click the **📁 Files** icon.
2. Click **Upload** and select your PDF file.
3. The file will appear at `/content/your-file.pdf`.
4. Update the path in the **Load Document** cell:

```python
load_pdf = PyPDFLoader('/content/your-file.pdf')
```

> Session storage is temporary. The uploaded file is lost when the runtime disconnects. Re-upload it on each new session, or mount Google Drive for persistent access (see [Future Improvements](#future-improvements)).

---

## Configuration

All tunable parameters are centralised in the **Config** cell:

```python
cohere_embed_model = 'embed-english-v3.0'   # Embedding model
cohere_chat_model  = 'command-r-08-2024'    # Generation model
qdrant_collection  = 'RAG_web-pdf'          # Vector collection name

CHUNK_SIZE    = 1000   # Maximum characters per chunk
CHUNK_OVERLAP = 200    # Overlapping characters between consecutive chunks
TOP_K         = 4      # Number of chunks to retrieve per query
```

**Tuning guidance:**

- Increase `CHUNK_SIZE` for documents with long, self-contained paragraphs (e.g. legal text, academic papers).
- Increase `CHUNK_OVERLAP` if answers seem to lose context at chunk boundaries.
- Increase `TOP_K` for broad questions that span multiple topics; reduce it for precise factual lookups to minimise noise.

---

## Pipeline Walkthrough

### 1. Document Ingestion

Two loaders are used and their outputs are merged into a single list:

```python
# Web pages
load_url = WebBaseLoader(web_paths=[
    'https://docs.cohere.com/docs/embeddings',
    'https://docs.langchain.com/oss/python/langchain/retrieval'
])

# PDF file
load_pdf = PyPDFLoader('/content/Chapter_4_v9.0[1].pdf')

# Unified document list
load_docs = load_url.load() + load_pdf.load()
```

`WebBaseLoader` uses `beautifulsoup4` and `lxml` under the hood to parse HTML and strip markup. `PyPDFLoader` extracts text page-by-page, preserving metadata such as `source` and `page`.

### 2. Text Chunking

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
splits = text_splitter.split_documents(load_docs)
```

`RecursiveCharacterTextSplitter` tries to split on natural boundaries (`\n\n`, `\n`, ` `) before falling back to hard character cuts. This produces semantically coherent chunks rather than arbitrary slices.

### 3. Embedding Generation

```python
embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv('COHERE_API_KEY'),
    model=cohere_embed_model
)
```

Cohere's `embed-english-v3.0` produces 1024-dimensional dense vectors optimised for English semantic similarity. The vector dimensionality is detected dynamically at runtime by embedding a test string and checking `len(test_vector)`, so the Qdrant collection is always created with the correct size.

### 4. Vector Storage

The pipeline checks whether the target collection already exists in Qdrant before indexing:

```python
if qdrant_collection not in existing_collections:
    # First run — create collection and index all chunks
    qdrant.create_collection(
        collection_name=qdrant_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    vectorStore = QdrantVectorStore.from_documents(...)
else:
    # Subsequent runs — connect to existing collection, skip re-indexing
    vectorStore = QdrantVectorStore.from_existing_collection(...)
```

This prevents duplicate indexing on re-runs — a common source of inflated vector stores in naive implementations. Cosine similarity is used as the distance metric, which is standard for text embeddings because it is invariant to vector magnitude.

### 5. Retrieval & Generation

```python
def queryRag(question: str):
    # Retrieve top-K semantically similar chunks
    docs = vectorStore.similarity_search(query=question, k=TOP_K)

    # Assemble context
    context = '\n\n'.join([doc.page_content for doc in docs])

    # Build grounded prompt
    prompt = f"""Based on the following context, answer the question clearly and concisely.

Context:
{context}

Question: {question}

Answer:"""

    # Generate answer
    response = llm.invoke(prompt)
    return response.content.strip()
```

The prompt is intentionally minimal — it instructs the model to answer only from the provided context. There is no system prompt telling the model to use general knowledge, which keeps answers grounded and auditable.

---

## Usage

### Running the Notebook

Run all cells **top to bottom** in order. Each section depends on the one before it.

1. **Install Libraries** — installs all dependencies into the Colab runtime
2. **Import Libraries** — loads all modules
3. **Config** — sets model names, collection name, and chunking parameters
4. **Load Document** — fetches web pages and reads the PDF from `/content/`
5. **Create Vector** — generates embeddings and sets up the Qdrant collection
6. **Building RAG System** — defines `queryRag()` and runs example queries



