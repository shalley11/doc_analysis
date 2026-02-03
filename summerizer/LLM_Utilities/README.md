# LLM_Utilities

A production-ready Python module for LLM interactions with comprehensive logging, context tracking, and iterative refinement support.

## Features

- **Multi-Backend Support**: Ollama and VLLM backends
- **Context Length Tracking**: Model-wise context limits with warnings and rejection
- **Refinement Cycle**: Iteratively improve outputs with user feedback
- **Regeneration**: Start fresh from original text with new instructions
- **Comprehensive Logging**: Structured logs with request tracing and metrics
- **Redis Session Storage**: Persistent refinement sessions with TTL

---

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `requests` - HTTP client for LLM API calls
- `pydantic` - Data validation
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `redis` - Session storage

### Prerequisites
- Redis server running on `localhost:6379`
- Ollama running on `localhost:11434` (or VLLM on `localhost:8000`)

---

## Quick Start

### Run the API Server

```bash
cd LLM_Utilities
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Access Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Configuration

### `config.py`

```python
# Backend selection
LLM_BACKEND = "ollama"  # or "vllm"

# Service URLs
OLLAMA_URL = "http://localhost:11434"
VLLM_URL = "http://localhost:8000"

# Default model
DEFAULT_MODEL = "gemma3:4b"

# Model context lengths
MODEL_CONTEXT_LENGTHS = {
    "gemma3:4b": 8192,   # POC
    "gemma3:12b": 8192,  # PROD
}

# Context thresholds (percentage)
CONTEXT_WARNING_THRESHOLD = 80   # Log WARNING
CONTEXT_ERROR_THRESHOLD = 95     # Log ERROR
CONTEXT_REJECT_THRESHOLD = 100   # Reject request
```

---

## API Endpoints

### POST `/process-text`
Process text with various tasks. Returns a `request_id` for refinement/regeneration.

**Request:**
```json
{
  "text": "Your long document here...",
  "task": "summary",
  "summary_type": "brief",
  "model": "gemma3:4b",
  "user_id": "user123"
}
```

**Tasks:** `summary`, `translate`, `rephrase`, `remove_repetitions`

**Summary Types:** `brief`, `detailed`, `bulletwise`

**Response:**
```json
{
  "request_id": "abc-123-uuid",
  "task": "summary",
  "model": "gemma3:4b",
  "output": "Generated summary...",
  "user_id": "user123"
}
```

---

### POST `/refine/{request_id}`
Refine the **current result** based on user feedback. Uses smaller context.

**Request:**
```json
{
  "user_feedback": "Make it shorter and add bullet points",
  "user_id": "user123"
}
```

**Response:**
```json
{
  "request_id": "abc-123-uuid",
  "refined_output": "Refined result...",
  "refinement_count": 1,
  "task": "summary",
  "model": "gemma3:4b",
  "user_id": "user123"
}
```

---

### POST `/regenerate/{request_id}`
Regenerate from **original text** with new instructions. Uses larger context but starts fresh.

**Request:**
```json
{
  "user_feedback": "Focus only on financial aspects",
  "user_id": "user123"
}
```

**Response:**
```json
{
  "request_id": "abc-123-uuid",
  "regenerated_output": "New result from original...",
  "regeneration_count": 1,
  "task": "summary",
  "model": "gemma3:4b",
  "user_id": "user123"
}
```

---

### GET `/refine/{request_id}`
Get current session status.

**Response:**
```json
{
  "request_id": "abc-123-uuid",
  "task": "summary",
  "model": "gemma3:4b",
  "current_output": "Latest output...",
  "refinement_count": 2,
  "regeneration_count": 1,
  "user_id": "user123",
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:35:00",
  "ttl_seconds": 3200
}
```

---

### DELETE `/refine/{request_id}`
End session and cleanup Redis data.

---

### POST `/refine/{request_id}/extend`
Extend session TTL.

**Query Parameter:** `ttl_seconds` (default: 3600)

---

### GET `/health`
Health check endpoint.

---

### GET `/logs/stats`
Get log file statistics.

---

## Refine vs Regenerate

| Aspect | `/refine` | `/regenerate` |
|--------|-----------|---------------|
| **Input** | `current_result` | `original_text` |
| **Context Size** | Smaller | Larger |
| **Use Case** | Polish, tweak, minor fixes | Start fresh, change direction |
| **Counter** | `refinement_count` | `regeneration_count` |

### When to Use Refine
- Output is mostly good, needs small improvements
- "Make it shorter", "Fix grammar", "Add conclusion"

### When to Use Regenerate
- Output went in wrong direction
- Need completely different focus
- "Focus on X instead of Y", "Ignore section Z"

---

## Context Length Protection

The API validates context length before processing:

| Usage | Action |
|-------|--------|
| < 80% | Normal processing |
| 80-95% | WARNING logged |
| 95-100% | ERROR logged |
| >= 100% | Request REJECTED (HTTP 400) |

**Error Response:**
```json
{
  "detail": {
    "error": "context_length_exceeded",
    "message": "Input text is too long and would be truncated.",
    "model": "gemma3:4b",
    "context_limit": 8192,
    "estimated_tokens": 9500,
    "usage_percent": 115.97
  }
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                           │
│                         (main.py)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ /process-   │    │  /refine    │    │    /regenerate      │ │
│  │    text     │    │             │    │                     │ │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘ │
│         │                  │                      │             │
│         ▼                  ▼                      ▼             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Context Validation                        ││
│  │              (config.py - estimate_tokens)                  ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                  │                      │             │
│         ▼                  ▼                      ▼             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Prompt Builder                           ││
│  │                     (prompts.py)                            ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                  │                      │             │
│         ▼                  ▼                      ▼             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     LLM Client                              ││
│  │    (llm_client.py - Ollama/VLLM with context logging)      ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                  │                      │             │
│         ▼                  ▼                      ▼             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Refinement Store                           ││
│  │           (refinement_store.py - Redis)                     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Flow Diagrams

### Initial Processing Flow
```
User Request
     │
     ▼
POST /process-text
     │
     ├─► Build Prompt (prompts.py)
     │
     ├─► Check Context Length
     │        │
     │        ├─► >= 100%? ──► HTTP 400 Error
     │        │
     │        └─► < 100%? ──► Continue
     │
     ├─► Call LLM (llm_client.py)
     │        │
     │        └─► Log context usage, metrics
     │
     ├─► Store in Redis (refinement_store.py)
     │        │
     │        └─► TTL: 1 hour
     │
     └─► Return response with request_id
```

### Refinement Flow
```
POST /refine/{request_id}
     │
     ├─► Fetch from Redis (current_result)
     │
     ├─► Build Refinement Prompt
     │        │
     │        └─► current_result + user_feedback
     │
     ├─► Check Context Length
     │
     ├─► Call LLM
     │
     ├─► Update Redis (overwrite current_result)
     │        │
     │        └─► Increment refinement_count
     │
     └─► Return refined output
```

### Regeneration Flow
```
POST /regenerate/{request_id}
     │
     ├─► Fetch from Redis (original_text)
     │
     ├─► Build Regeneration Prompt
     │        │
     │        └─► original_text + user_feedback
     │
     ├─► Check Context Length
     │
     ├─► Call LLM
     │
     ├─► Update Redis (overwrite current_result)
     │        │
     │        └─► Increment regeneration_count
     │
     └─► Return regenerated output
```

---

## Logging

### Log Files (in `logs/` directory)

| File | Content |
|------|---------|
| `llm_requests.log` | All LLM API calls |
| `llm_errors.log` | Error-level logs |
| `llm_metrics.log` | Performance metrics (JSON) |
| `llm_debug.log` | Detailed debug info |

### Log Format
```
2024-01-15 10:30:00 | INFO     | abc-123-uuid | user123 | LLM_REQUEST | model=gemma3:4b | task=summary
2024-01-15 10:30:05 | WARNING  | abc-123-uuid | user123 | CONTEXT_USAGE_HIGH | model=gemma3:4b | usage=85.2%
2024-01-15 10:30:10 | INFO     | abc-123-uuid | user123 | LLM_RESPONSE | status=success | latency=5000ms
```

### Context Usage Logs
```
# Normal
DEBUG | CONTEXT_USAGE | model=gemma3:4b | usage=25.0% | tokens=500/8192

# Warning (>80%)
WARNING | CONTEXT_USAGE_HIGH | model=gemma3:4b | usage=85.2% | tokens=6980/8192

# Error (>95%)
ERROR | CONTEXT_NEAR_LIMIT | model=gemma3:4b | usage=96.1% | tokens=7872/8192

# Critical (>=100%)
CRITICAL | CONTEXT_EXCEEDED | model=gemma3:4b | usage=105.3% | tokens=8626/8192
```

---

## Python SDK Usage

```python
from LLM_Utilities import (
    generate_text,
    generate_text_with_logging,
    stream_text,
    setup_llm_logging,
    get_model_context_length,
    estimate_tokens
)

# Initialize logging
setup_llm_logging()

# Simple generation
result = generate_text("Summarize this text...")

# Generation with logging
result = generate_text_with_logging(
    prompt="Summarize this text...",
    model="gemma3:4b",
    task="summary"
)

# Streaming
for token in stream_text("Summarize this text..."):
    print(token, end='', flush=True)

# Check context before calling
prompt = "Your long prompt..."
model = "gemma3:4b"
context_limit = get_model_context_length(model)
tokens = estimate_tokens(prompt)
usage = (tokens / context_limit) * 100

if usage < 100:
    result = generate_text(prompt)
else:
    print(f"Prompt too long: {usage:.1f}% of context")
```

---

## File Structure

```
LLM_Utilities/
├── __init__.py           # Package exports
├── config.py             # Configuration & context limits
├── llm_client.py         # LLM backend integration
├── logging_config.py     # Structured logging system
├── prompts.py            # Prompt templates
├── refinement_store.py   # Redis session storage
├── schemas.py            # Pydantic models
├── main.py               # FastAPI application
├── requirements.txt      # Dependencies
├── README.md             # This documentation
└── logs/                 # Log files (auto-created)
    ├── llm_requests.log
    ├── llm_errors.log
    ├── llm_metrics.log
    └── llm_debug.log
```

---

## Adding New Models

Update `config.py`:

```python
MODEL_CONTEXT_LENGTHS = {
    "gemma3:4b": 8192,
    "gemma3:12b": 8192,
    "llama3:8b": 8192,
    "mistral:7b": 32768,  # Add new model
}
```

---

## Error Handling

| HTTP Code | Reason |
|-----------|--------|
| 400 | Context length exceeded |
| 404 | Request ID not found or expired |
| 500 | Internal server error |

---

## Session Management

- **TTL**: 2 hours default
- **Extend**: `POST /refine/{id}/extend?ttl_seconds=3600`
- **Cleanup**: `DELETE /refine/{id}` or auto-expire

---

## Version History

| Version | Changes |
|---------|---------|
| 2.1.0 | Added `/regenerate` endpoint, context length protection |
| 2.0.0 | Added refinement cycle with Redis storage |
| 1.0.0 | Initial release with basic LLM integration |
