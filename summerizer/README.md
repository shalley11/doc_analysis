# PDF Summarizer

A comprehensive PDF processing and analysis system with structure-aware chunking, vision model integration, and on-demand summarization.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PDF SUMMARIZER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Web UI    │    │  REST API   │    │   Milvus    │    │   Ollama    │  │
│  │  (Port 80)  │───▶│ (Port 8080) │───▶│(Port 19530) │    │(Port 11434) │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                            │                   │                  │         │
│                            ▼                   │                  │         │
│                     ┌─────────────┐            │                  │         │
│                     │ RQ Workers  │────────────┼──────────────────┘         │
│                     │   (Redis)   │            │                            │
│                     └─────────────┘            │                            │
│                            │                   │                            │
│                            ▼                   │                            │
│                     ┌─────────────┐            │                            │
│                     │  Embedding  │────────────┘                            │
│                     │  (Port 8000)│                                         │
│                     └─────────────┘                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Processing Pipeline

### 1. PDF Upload & Processing Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PDF PROCESSING PIPELINE                              │
└──────────────────────────────────────────────────────────────────────────────┘

  ┌─────────┐
  │  User   │
  │ Upload  │
  └────┬────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Web UI     │────▶│  FastAPI     │────▶│  RQ Queue    │
│              │     │  /api/v2/    │     │  (Redis)     │
└──────────────┘     │  pdf/analyze │     └──────┬───────┘
                     └──────────────┘            │
                                                 ▼
                     ┌───────────────────────────────────────┐
                     │           RQ Worker Process           │
                     ├───────────────────────────────────────┤
                     │                                       │
                     │  ┌─────────────────────────────────┐  │
                     │  │ 1. PDF Extraction (PyMuPDF)     │  │
                     │  │    - Text blocks with fonts     │  │
                     │  │    - Images extraction          │  │
                     │  │    - Table detection            │  │
                     │  └───────────────┬─────────────────┘  │
                     │                  ▼                    │
                     │  ┌─────────────────────────────────┐  │
                     │  │ 2. Structure Detection          │  │
                     │  │    - Font analysis              │  │
                     │  │    - Heading detection (H1-H3)  │  │
                     │  │    - Section hierarchy          │  │
                     │  └───────────────┬─────────────────┘  │
                     │                  ▼                    │
                     │  ┌─────────────────────────────────┐  │
                     │  │ 3. Vision Processing (Gemma3)   │  │
                     │  │    - Image batch processing     │  │
                     │  │    - Table batch processing     │  │
                     │  │    - Generate summaries/captions│  │
                     │  └───────────────┬─────────────────┘  │
                     │                  ▼                    │
                     │  ┌─────────────────────────────────┐  │
                     │  │ 4. Structure-Aware Chunking     │  │
                     │  │    - Paragraph chunks (≤500w)   │  │
                     │  │    - Table chunks with summary  │  │
                     │  │    - Image chunks with caption  │  │
                     │  │    - List chunks                │  │
                     │  │    - Section context prefix     │  │
                     │  └───────────────┬─────────────────┘  │
                     │                  ▼                    │
                     │  ┌─────────────────────────────────┐  │
                     │  │ 5. Embedding Generation         │  │
                     │  │    - E5-Large (1024 dim)        │  │
                     │  │    - Batch processing           │  │
                     │  └───────────────┬─────────────────┘  │
                     │                  ▼                    │
                     │  ┌─────────────────────────────────┐  │
                     │  │ 6. Milvus Indexing              │  │
                     │  │    - Store chunks + metadata    │  │
                     │  │    - Vision metadata fields     │  │
                     │  └─────────────────────────────────┘  │
                     │                                       │
                     └───────────────────────────────────────┘
```

### 2. Chunk Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MILVUS CHUNK SCHEMA                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Core Fields:                        Vision Metadata:                       │
│  ├── chunk_id (VARCHAR)              ├── table_summary (VARCHAR)            │
│  ├── embedding (FLOAT_VECTOR[1024])  ├── image_caption (VARCHAR)            │
│  ├── text (VARCHAR)                  └── image_summary (VARCHAR)            │
│  ├── content_type (VARCHAR)                                                 │
│  ├── pdf_name (VARCHAR)              Structure Fields:                      │
│  ├── page_no (INT64)                 ├── section_hierarchy (VARCHAR/JSON)   │
│  ├── position (FLOAT)                └── heading_level (INT64)              │
│  ├── chunk_number (INT64)                                                   │
│  ├── image_link (VARCHAR)            Context Links:                         │
│  ├── table_link (VARCHAR)            ├── context_before_id (VARCHAR)        │
│  └── session_id (VARCHAR)            └── context_after_id (VARCHAR)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Vision Model Batch Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     VISION MODEL BATCH PROCESSING                            │
└─────────────────────────────────────────────────────────────────────────────┘

  Page Blocks
       │
       ▼
┌──────────────────┐
│ Separate by Type │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│Images │ │Tables │
│ List  │ │ List  │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│   IMAGE BATCH (size=5)  │  │   TABLE BATCH (size=5)  │
├─────────────────────────┤  ├─────────────────────────┤
│                         │  │                         │
│  Prompt: IMAGE_BATCH_   │  │  Prompt: TABLE_BATCH_   │
│  SUMMARY_PROMPT         │  │  SUMMARY_PROMPT         │
│  ┌───────────────────┐  │  │  ┌───────────────────┐  │
│  │ - Image type      │  │  │  │ - Table purpose   │  │
│  │ - Main subject    │  │  │  │ - Column headers  │  │
│  │ - Key data        │  │  │  │ - Key values      │  │
│  │ - Text/labels     │  │  │  │ - Patterns        │  │
│  │ - Insight         │  │  │  │ - Totals          │  │
│  └───────────────────┘  │  │  └───────────────────┘  │
│                         │  │                         │
│  Output:                │  │  Output:                │
│  - image_summary        │  │  - table_summary        │
│  - image_caption        │  │                         │
│                         │  │                         │
└─────────────────────────┘  └─────────────────────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
              ┌───────────────┐
              │ Updated Blocks│
              │ with Metadata │
              └───────────────┘
```

### 4. On-Demand Summary Generation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ON-DEMAND SUMMARY GENERATION                              │
└─────────────────────────────────────────────────────────────────────────────┘

                         User Request
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │  Document   │  │   Corpus    │  │    List     │
     │  Summary    │  │  Summary    │  │  Documents  │
     │   (POST)    │  │   (POST)    │  │    (GET)    │
     └──────┬──────┘  └──────┬──────┘  └─────────────┘
            │                │
            ▼                ▼
     ┌─────────────┐  ┌─────────────────────────┐
     │ Fetch Chunks│  │ For each document:      │
     │ from Milvus │  │   Fetch chunks          │
     │ (filter by  │  │   Generate doc summary  │
     │  pdf_name)  │  └───────────┬─────────────┘
     └──────┬──────┘              │
            │                     ▼
            │              ┌─────────────┐
            │              │  Aggregate  │
            │              │  Summaries  │
            │              └──────┬──────┘
            │                     │
            ▼                     ▼
     ┌─────────────────────────────────────┐
     │         Gemma3 Model (Ollama)       │
     ├─────────────────────────────────────┤
     │                                     │
     │  Summary Types:                     │
     │  ┌─────────────────────────────┐    │
     │  │ brief    │ 3-5 sentences    │    │
     │  │ detailed │ Full structured  │    │
     │  │ bullets  │ Key points list  │    │
     │  └─────────────────────────────┘    │
     │                                     │
     └─────────────────┬───────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  JSON Response  │
              │  with Summary   │
              └─────────────────┘
```

### 5. Q&A Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Q&A PIPELINE                                       │
└─────────────────────────────────────────────────────────────────────────────┘

  User Question
       │
       ▼
┌──────────────┐
│  Embedding   │──────────────────┐
│  Service     │                  │
└──────────────┘                  │
       │                          │
       ▼                          ▼
┌──────────────┐          ┌──────────────┐
│   Milvus     │          │  Query       │
│   Search     │◀─────────│  Embedding   │
└──────┬───────┘          └──────────────┘
       │
       ▼
┌──────────────┐
│  Top-K       │
│  Relevant    │
│  Chunks      │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│            LLM Generator             │
├──────────────────────────────────────┤
│  Providers:                          │
│  - Ollama (local)                    │
│  - OpenAI                            │
│  - Anthropic                         │
│  - Google Gemini                     │
└──────────────────┬───────────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Answer with     │
          │ Citations       │
          │ [Source: PDF,   │
          │  Page X]        │
          └─────────────────┘
```

## Module Structure

```
summerizer/
├── document_analysis_main.py    # FastAPI application & endpoints
├── run_services.sh              # Service startup script
│
├── pdf/                         # PDF Processing
│   ├── pdf_utils.py            # PDF extraction, page processing
│   ├── vision_utils.py         # Vision models, batch processing
│   └── structure_detector.py   # Font analysis, heading detection
│
├── chunking/                    # Text Chunking
│   ├── chunking_utils.py       # Structure-aware chunking
│   └── chunk_indexer.py        # Milvus indexing
│
├── embedding/                   # Embedding Service
│   ├── embedding_service.py    # E5-Large embedding server
│   └── embedding_client.py     # Embedding API client
│
├── vector_store/               # Vector Database
│   └── milvus_store.py        # Milvus operations
│
├── jobs/                       # Background Processing
│   ├── jobs.py                # PDF batch processing jobs
│   ├── job_state.py           # Job state management
│   └── processing_status.py   # Status tracking
│
├── qa/                         # Question Answering
│   ├── qa_service.py          # Q&A orchestration
│   ├── retriever.py           # Chunk retrieval
│   ├── generator.py           # LLM generators
│   ├── prompts.py             # Q&A prompts
│   └── summary_service.py     # On-demand summarization
│
└── ui/                         # Web Interface
    ├── index.html
    ├── styles.css
    └── app.js
```

## API Endpoints

### PDF Processing
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v2/pdf/analyze` | Upload and process PDFs |
| GET | `/api/v2/status/{batch_id}` | Get processing status |
| GET | `/api/v2/chunks/{batch_id}` | Get processed chunks |

### Summary Generation (On-Demand)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v2/summary/documents/{batch_id}` | List documents for summary |
| POST | `/api/v2/summary/document/{batch_id}/{pdf_name}` | Generate document summary |
| POST | `/api/v2/summary/corpus/{batch_id}` | Generate corpus summary |

### Q&A
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v2/qa/ask/{batch_id}` | Ask a question |
| POST | `/api/v2/qa/chat/{batch_id}` | Multi-turn chat |
| GET | `/api/v2/qa/search/{batch_id}` | Semantic search |

## Configuration

### Environment Variables

```bash
# Vision Model
USE_GEMMA3=true                    # Enable Gemma3 vision model
GEMMA3_MODE=local                  # "local" (Ollama) or "api"
VISION_BATCH_SIZE=5                # Batch size for vision processing

# Summary Generation
SUMMARY_MODEL=gemma3:4b            # Model for summarization
SUMMARY_TIMEOUT=1800               # Timeout in seconds (30 min)

# Services
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_SERVICE_URL=http://localhost:8000
MILVUS_HOST=localhost
MILVUS_PORT=19530

# LLM Providers (optional)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

## Quick Start

```bash
# 1. Start required services
docker-compose up -d milvus redis

# 2. Start Ollama with Gemma3
ollama pull gemma3:4b
ollama serve

# 3. Start embedding service
python -m uvicorn embedding.embedding_service:app --port 8000 &

# 4. Start RQ worker
USE_GEMMA3=true rq worker pdf-processing &

# 5. Start API server
USE_GEMMA3=true python -m uvicorn document_analysis_main:app --port 8080

# 6. Open UI
open http://localhost:8080/ui/
```

## Features

- **Structure-Aware Chunking**: Preserves document hierarchy with section context
- **Vision Model Integration**: Gemma3 for table/image analysis with batch processing
- **On-Demand Summarization**: Generate summaries when needed, not during processing
- **Multi-Provider LLM**: Support for Ollama, OpenAI, Anthropic, Gemini
- **Semantic Search**: E5-Large embeddings with Milvus vector store
- **Real-time Status**: WebSocket updates during processing
