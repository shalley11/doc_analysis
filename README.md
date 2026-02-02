# Document Analysis API

A comprehensive PDF processing and AI-powered summarization system that extracts, chunks, embeds, and indexes document content for semantic search and intelligent summarization.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [WebSocket Events](#websocket-events)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

## Overview

The Document Analysis API provides a complete pipeline for processing PDF documents:

1. **Upload** - Accept multiple PDFs (up to 5 per batch)
2. **Extract** - Extract text, tables, and images from each page
3. **Chunk** - Semantically chunk content with overlap for context preservation
4. **Embed** - Generate 1024-dimensional embeddings using E5-large model
5. **Index** - Store chunks in Milvus vector database for similarity search
6. **Summarize** - Generate AI-powered summaries using hierarchical map-reduce

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Client                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Upload    │  │   Status    │  │  Summary    │  │  WebSocket  │    │
│  │  Endpoint   │  │  Endpoint   │  │  Endpoints  │  │  Endpoints  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │    Redis    │ │   Milvus    │ │   Ollama    │
            │  Job Queue  │ │ Vector DB   │ │ Vision/LLM  │
            │   Pub/Sub   │ │             │ │             │
            └─────────────┘ └─────────────┘ └─────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          RQ Worker Process                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │    PDF      │  │  Chunking   │  │   Vision    │  │  Embedding  │    │
│  │ Extraction  │  │   Engine    │  │  Processing │  │  Generator  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
PDF Upload → Extract (pdfplumber/PyMuPDF) → Chunk (semantic) → Vision (Ollama)
                                                    │
                                                    ▼
                                           Embed (E5-large)
                                                    │
                                                    ▼
                                           Index (Milvus)
                                                    │
                                                    ▼
                                         Summarize (LLM) → Cache (Redis)
```

## Features

### PDF Processing
- Multi-PDF batch upload (up to 5 PDFs)
- Page range selection for partial processing
- Text, table, and image extraction
- Scanned PDF detection
- Multi-column layout handling
- Boilerplate content filtering

### Semantic Chunking
- Word-based chunking with configurable limits
- Overlap for context preservation
- Table-aware chunking (preserves headers)
- Deterministic chunk IDs (SHA256)

### Embedding & Indexing
- E5-large-v2 embeddings (1024 dimensions)
- Batch processing for efficiency
- Milvus vector database storage
- Chunk navigation (prev/next linking)

### Vision Processing
- Table image analysis
- Document image understanding
- Parallel batch processing
- Ollama integration (Gemma3)

### Summarization
- Multiple summary types (brief, bulletwise, detailed, executive)
- Hierarchical map-reduce for large documents
- Direct summarization for small documents
- Summary caching (Redis + Milvus hybrid)
- Interactive refinement with user feedback

### Real-time Updates
- WebSocket connections for live progress
- Redis Pub/Sub event distribution
- Separate channels for PDF and summary updates

## Installation

### Prerequisites

- Python 3.10+
- Redis Server
- Milvus Vector Database
- Ollama (for vision and LLM models)

### Setup

1. **Clone the repository**
   ```bash
   git clone git@github.com:shalley11/doc_analysis.git
   cd doc_analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start required services**
   ```bash
   # Redis
   redis-server

   # Milvus (using Docker)
   docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest

   # Ollama
   ollama serve
   ollama pull gemma3:4b
   ```

4. **Start the RQ worker**
   ```bash
   python -m doc_analysis.workers.rq_worker
   ```

5. **Start the API server**
   ```bash
   uvicorn doc_analysis.api:app --host 0.0.0.0 --port 8000
   ```

## Configuration

Configuration is managed in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_PDFS_PER_BATCH` | 5 | Maximum PDFs per upload |
| `BATCH_TTL_SECONDS` | 86400 | Data retention (24 hours) |
| `EMBEDDING_DIM` | 1024 | E5-large embedding dimension |
| `CHUNK_MIN_WORDS` | 50 | Minimum words per chunk |
| `CHUNK_MAX_WORDS` | 500 | Maximum words per chunk |
| `CHUNK_OVERLAP_WORDS` | 50 | Overlap between chunks |
| `VISION_BATCH_SIZE` | 3 | Parallel vision processing |
| `VISION_TIMEOUT` | 900 | Vision processing timeout (15 min) |
| `SUMMARY_MODEL` | gemma3:4b | Ollama model for summarization |
| `SUMMARY_STORAGE_MODE` | hybrid | Redis intermediate, Milvus final |

### Environment Variables

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
MILVUS_HOST=localhost
MILVUS_PORT=19530
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

## Usage

### Upload PDFs

```bash
curl -X POST "http://localhost:8000/upload-pdfs" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -F "start_page=1" \
  -F "end_page=50"
```

Response:
```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "pdf_count": 2,
  "status": "queued",
  "page_range": {"start": 1, "end": 50}
}
```

### Check Job Status

```bash
curl "http://localhost:8000/job-status/550e8400-e29b-41d4-a716-446655440000"
```

### Generate Summary

```bash
# Single PDF summary
curl "http://localhost:8000/summary/pdf?batch_id=550e8400&pdf_name=document1.pdf&summary_type=brief"

# All PDFs combined summary
curl "http://localhost:8000/summary/all?batch_id=550e8400&summary_type=detailed"
```

### Request-Based Summary Workflow (Recommended)

The request-based workflow tracks each summary with a unique `request_id` for easy refinement and regeneration.

#### Step 1: Generate Summary with Request ID

```bash
curl "http://localhost:8000/summary/generate?batch_id=550e8400&pdf_name=document1.pdf&summary_type=detailed"
```

Response:
```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "pdf_name": "document1.pdf",
  "summary": "This document provides...",
  "method": "hierarchical"
}
```

#### Step 2: Refine Summary (Using Previous Summary Only)

```bash
curl -X POST "http://localhost:8000/summary/request/refine" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "user_feedback": "Make it more concise and use bullet points"
  }'
```

#### Step 3: Regenerate Summary (Fetches from Milvus)

```bash
curl -X POST "http://localhost:8000/summary/request/regenerate" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "user_feedback": "Focus on financial metrics and risk factors",
    "top_k": 20
  }'
```

#### View Request History

```bash
curl "http://localhost:8000/summary/history/550e8400?pdf_name=document1.pdf&limit=10"
```

### Legacy Refine Summary

```bash
curl -X POST "http://localhost:8000/summary/refine/contextual" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "550e8400-e29b-41d4-a716-446655440000",
    "pdf_name": "document1.pdf",
    "summary_type": "detailed",
    "user_feedback": "Include more details about financial projections",
    "top_k": 15
  }'
```

### WebSocket Connection

```javascript
// PDF processing updates
const ws = new WebSocket('ws://localhost:8000/ws/pdf/550e8400-e29b-41d4-a716-446655440000');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data.event, 'Data:', data);
};

// Summarization updates
const summaryWs = new WebSocket('ws://localhost:8000/ws/summary/550e8400-e29b-41d4-a716-446655440000');
```

## API Reference

Access interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/upload-pdfs` | Upload PDF documents |
| GET | `/job-status/{batch_id}` | Get processing status |
| GET | `/summary/pdfs/{batch_id}` | List PDFs in batch |
| GET | `/summary/pdf` | Generate PDF summary |
| GET | `/summary/all` | Generate combined summary |
| GET | `/summary/generate` | Generate summary with request_id |
| POST | `/summary/request/refine` | Refine by request_id + feedback |
| POST | `/summary/request/regenerate` | Regenerate from Milvus + feedback |
| GET | `/summary/request/{id}` | Get request details |
| DELETE | `/summary/request/{id}` | Delete request |
| GET | `/summary/history/{batch_id}` | Get request history |
| POST | `/summary/refine/simple` | Legacy: Refine (feedback only) |
| POST | `/summary/refine/contextual` | Legacy: Refine (with context) |
| GET | `/ws/stats` | WebSocket statistics |
| WS | `/ws/pdf/{batch_id}` | PDF processing updates |
| WS | `/ws/summary/{batch_id}` | Summarization updates |

## WebSocket Events

### PDF Processing Events

| Event | Description |
|-------|-------------|
| `pdf.started` | Processing started |
| `pdf.extracting` | Extracting content |
| `pdf.extracted` | Extraction complete |
| `pdf.chunking` | Creating chunks |
| `pdf.chunked` | Chunking complete |
| `pdf.vision_processing` | Processing tables/images |
| `pdf.embedding` | Generating embeddings |
| `pdf.embedded` | Embedding complete |
| `pdf.completed` | All processing done |
| `pdf.failed` | Error occurred |

### Summarization Events

| Event | Description |
|-------|-------------|
| `summary.started` | Summarization started |
| `summary.cache_hit` | Using cached summary |
| `summary.method_selected` | Method chosen (direct/hierarchical) |
| `summary.batch_started` | Batch N/M started |
| `summary.batch_completed` | Batch N/M completed |
| `summary.reduce_started` | Combining summaries |
| `summary.reduce_level` | Reduce level progress |
| `summary.llm_call_started` | LLM API call started |
| `summary.llm_call_completed` | LLM API call done |
| `summary.completed` | Summary ready |
| `summary.failed` | Error occurred |

## Project Structure

```
doc_analysis/
├── api.py                    # FastAPI application
├── config.py                 # Configuration settings
├── logging_config.py         # Logging setup
├── requirements.txt          # Dependencies
│
├── chunking/                 # Document chunking
│   ├── __init__.py
│   └── chunk_builder.py      # Semantic chunking logic
│
├── embedding/                # Embedding generation
│   ├── __init__.py
│   └── e5_embedder.py        # E5-large embedder
│
├── jobs/                     # Job tracking
│   ├── __init__.py
│   └── job_store.py          # Redis-based job store
│
├── pdf/                      # PDF processing
│   ├── __init__.py
│   ├── text_table_extractor.py
│   ├── layout_extractor.py
│   ├── image_extractor.py
│   ├── scan_detector.py
│   ├── boilerplate_filter.py
│   ├── column_utils.py
│   ├── heading_inference.py
│   └── structure_merger.py
│
├── qa/                       # Q&A utilities
│   ├── __init__.py
│   └── rag_qa.py             # RAG-based Q&A
│
├── queues/                   # Job queue
│   └── enqueue.py            # RQ job enqueueing
│
├── realtime/                 # Real-time updates
│   ├── __init__.py
│   ├── events.py             # Event definitions
│   ├── event_publisher.py    # Redis Pub/Sub publisher
│   ├── event_subscriber.py   # Redis Pub/Sub subscriber
│   └── ws_manager.py         # WebSocket manager
│
├── summarization/            # Summary generation
│   ├── __init__.py
│   ├── hierarchical_summarizer.py
│   ├── summarizer.py
│   ├── summary_prompts.py
│   ├── summary_service.py
│   └── summary_store.py
│
├── vector_store/             # Vector database
│   ├── milvus_schema.py
│   ├── milvus_store.py
│   └── milvus_utils.py
│
├── vision/                   # Vision processing
│   ├── prompts.py
│   └── vision_worker.py
│
├── workers/                  # Background workers
│   ├── __init__.py
│   ├── ingestion_worker.py
│   ├── pdf_ingestion.py
│   └── rq_worker.py
│
└── cleanup/                  # Maintenance
    ├── __init__.py
    ├── cleanup_service.py
    └── run_cleanup.py
```

## Dependencies

### Core
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### PDF Processing
- **pdfplumber** - PDF text/table extraction
- **PyMuPDF** - PDF rendering and image extraction

### Machine Learning
- **transformers** - Hugging Face transformers
- **torch** - PyTorch for model inference
- **numpy** - Numerical operations

### Storage
- **pymilvus** - Milvus vector database client
- **redis** - Redis client for caching and queues
- **rq** - Redis Queue for job processing

### Other
- **requests** - HTTP client for Ollama API

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
