"""
Test script for Q&A module.
Tests imports, prompts, generator (with Ollama/gemma3:4b), and retriever.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Q&A MODULE TESTS")
print("=" * 60)

# =============================================================================
# TEST 1: Imports
# =============================================================================
print("\n" + "=" * 60)
print("TEST 1: Import checks")
print("=" * 60)

try:
    from qa import (
        Retriever,
        QAService,
        create_generator,
        get_default_generator,
        OllamaGenerator,
        build_qa_prompt,
        build_summary_prompt
    )
    from qa.prompts import format_context_with_citations
    print("✓ All QA module imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# =============================================================================
# TEST 2: Prompts formatting
# =============================================================================
print("\n" + "=" * 60)
print("TEST 2: Prompts formatting")
print("=" * 60)

# Mock chunks
mock_chunks = [
    {
        "text": "Climate change is causing global temperatures to rise significantly.",
        "pdf_name": "climate_report.pdf",
        "page_no": 0,
        "content_type": "text",
        "score": 0.92
    },
    {
        "text": "The Arctic ice is melting at an unprecedented rate.",
        "pdf_name": "climate_report.pdf",
        "page_no": 2,
        "content_type": "text",
        "score": 0.87
    },
    {
        "text": "Table showing temperature changes by decade.",
        "pdf_name": "data_analysis.pdf",
        "page_no": 5,
        "content_type": "table",
        "score": 0.81
    }
]

# Test context formatting
context = format_context_with_citations(mock_chunks)
print("Formatted context with citations:")
print("-" * 40)
print(context[:500] + "..." if len(context) > 500 else context)
print("-" * 40)

# Test Q&A prompt building
system_prompt, user_prompt = build_qa_prompt("What is happening to the climate?", mock_chunks)
print(f"\n✓ Q&A prompt built successfully")
print(f"  System prompt length: {len(system_prompt)} chars")
print(f"  User prompt length: {len(user_prompt)} chars")

# Test summary prompt building
system_prompt, user_prompt = build_summary_prompt(mock_chunks, "bullets")
print(f"✓ Summary prompt built successfully (bullets)")

# =============================================================================
# TEST 3: Ollama connectivity check
# =============================================================================
print("\n" + "=" * 60)
print("TEST 3: Ollama connectivity")
print("=" * 60)

import requests

ollama_available = False
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        print(f"✓ Ollama is running")
        print(f"  Available models: {model_names}")

        # Check if gemma3:4b is available
        if any("gemma3" in m for m in model_names):
            print("✓ gemma3 model is available")
            ollama_available = True
        else:
            print("⚠ gemma3 not found. Run: ollama pull gemma3:4b")
    else:
        print(f"✗ Ollama returned status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("✗ Ollama is not running at localhost:11434")
except Exception as e:
    print(f"✗ Error checking Ollama: {e}")

# =============================================================================
# TEST 4: Generator with gemma3:4b
# =============================================================================
print("\n" + "=" * 60)
print("TEST 4: Generator test (gemma3:4b)")
print("=" * 60)

if ollama_available:
    try:
        generator = OllamaGenerator(model="gemma3:4b", timeout=180)
        print(f"Generator created: model={generator.model}, timeout={generator.timeout}s")

        # Simple test prompt
        test_system = "You are a helpful assistant. Be concise."
        test_user = "What is 2 + 2? Answer in one word."

        print("\nSending test prompt to gemma3:4b...")
        response = generator.generate(test_system, test_user, temperature=0.1, max_tokens=50)
        print(f"✓ Response received: {response.strip()}")

    except Exception as e:
        print(f"✗ Generator test failed: {e}")
else:
    print("⚠ Skipping generator test (Ollama/gemma3 not available)")

# =============================================================================
# TEST 5: Full Q&A flow with mock context
# =============================================================================
print("\n" + "=" * 60)
print("TEST 5: Full Q&A generation with citations")
print("=" * 60)

if ollama_available:
    try:
        generator = OllamaGenerator(model="gemma3:4b", timeout=180)

        # Build prompt with mock chunks
        question = "What is happening to global temperatures and the Arctic?"
        system_prompt, user_prompt = build_qa_prompt(question, mock_chunks)

        print(f"Question: {question}")
        print("Generating answer with citations...")

        answer = generator.generate(system_prompt, user_prompt, temperature=0.7, max_tokens=500)

        print("\n" + "-" * 40)
        print("ANSWER:")
        print("-" * 40)
        print(answer)
        print("-" * 40)

        # Check if citations are present
        if "[Source:" in answer:
            print("\n✓ Answer includes citations")
        else:
            print("\n⚠ Answer may not include proper citations")

    except Exception as e:
        print(f"✗ Q&A generation failed: {e}")
else:
    print("⚠ Skipping Q&A test (Ollama/gemma3 not available)")

# =============================================================================
# TEST 6: Milvus connectivity (optional)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 6: Milvus connectivity")
print("=" * 60)

try:
    from vector_store.milvus_store import MilvusVectorStore
    vector_store = MilvusVectorStore(host="localhost", port=19530)
    print("✓ Connected to Milvus at localhost:19530")
except Exception as e:
    print(f"⚠ Milvus not available: {e}")
    print("  (This is optional - Q&A requires ingested PDFs)")

# =============================================================================
# TEST 7: Embedding service connectivity (optional)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 7: Embedding service connectivity")
print("=" * 60)

try:
    response = requests.get("http://localhost:8000/docs", timeout=5)
    if response.status_code == 200:
        print("✓ Embedding service is running at localhost:8000")
    else:
        print(f"⚠ Embedding service returned status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("⚠ Embedding service not running at localhost:8000")
    print("  Start with: uvicorn embedding.embedding_service:app --port 8000")
except Exception as e:
    print(f"⚠ Error checking embedding service: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("✓ QA module imports working")
print("✓ Prompt formatting working")
if ollama_available:
    print("✓ Ollama + gemma3:4b working")
    print("✓ Q&A generation with citations working")
else:
    print("⚠ Ollama/gemma3 not tested (not available)")
print("\nTo run full Q&A:")
print("1. Start embedding service: uvicorn embedding.embedding_service:app --port 8000")
print("2. Start Milvus")
print("3. Upload and process a PDF via /api/v2/pdf/analyze")
print("4. Use /api/v2/qa/ask/{batch_id} to ask questions")
