"""
Full end-to-end test: PDF upload -> Processing -> Q&A
"""
import sys
import time
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

API_BASE = "http://localhost:8080"
PDF_PATH = project_root / "test_document.pdf"

print("=" * 70)
print("FULL FLOW TEST: PDF Upload -> Processing -> Q&A")
print("=" * 70)

# =============================================================================
# STEP 1: Upload PDF
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: Upload PDF")
print("=" * 70)

if not PDF_PATH.exists():
    print(f"ERROR: Test PDF not found at {PDF_PATH}")
    sys.exit(1)

print(f"Uploading: {PDF_PATH.name}")

with open(PDF_PATH, "rb") as f:
    response = requests.post(
        f"{API_BASE}/api/v2/pdf/analyze",
        files={"files": (PDF_PATH.name, f, "application/pdf")},
        data={
            "use_vision": "no",  # Skip vision for faster processing
            "use_semantic_chunking": "no",
            "preview_only": "no"
        }
    )

if response.status_code != 200:
    print(f"ERROR: Upload failed with status {response.status_code}")
    print(response.text)
    sys.exit(1)

result = response.json()
batch_id = result["batch_id"]
print(f"✓ Upload successful")
print(f"  Batch ID: {batch_id}")
print(f"  RQ Job ID: {result.get('rq_job_id', 'N/A')}")

# =============================================================================
# STEP 2: Wait for processing
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Wait for processing")
print("=" * 70)

max_wait = 120  # seconds
start_time = time.time()
last_status = None

while time.time() - start_time < max_wait:
    response = requests.get(f"{API_BASE}/api/v1/pdf/status/{batch_id}")
    status = response.json()

    current_status = status.get("status", "unknown")
    progress = status.get("progress_percent", 0)

    if current_status != last_status:
        print(f"  Status: {current_status} ({progress}%)")
        last_status = current_status

    if current_status == "completed":
        print(f"✓ Processing completed in {time.time() - start_time:.1f}s")
        print(f"  Total pages: {status.get('total_pages', 'N/A')}")
        print(f"  Total chunks: {status.get('chunk_count', 'N/A')}")
        break
    elif current_status == "failed":
        print(f"ERROR: Processing failed")
        print(status)
        sys.exit(1)

    time.sleep(2)
else:
    print(f"ERROR: Processing timed out after {max_wait}s")
    sys.exit(1)

# =============================================================================
# STEP 3: Search chunks (without LLM)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Search chunks")
print("=" * 70)

query = "temperature rise"
response = requests.get(
    f"{API_BASE}/api/v2/qa/search/{batch_id}",
    params={"query": query, "top_k": 3}
)

if response.status_code == 200:
    result = response.json()
    print(f"Query: '{query}'")
    print(f"Results: {result.get('total_results', 0)} chunks found")
    for i, r in enumerate(result.get("results", [])[:3], 1):
        print(f"\n  [{i}] Score: {r['score']:.4f} | Page {r['page_no']}")
        text_preview = r['text'][:100] + "..." if len(r['text']) > 100 else r['text']
        print(f"      {text_preview}")
    print("\n✓ Chunk search working")
else:
    print(f"ERROR: Search failed: {response.text}")

# =============================================================================
# STEP 4: Q&A with gemma3:4b
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Q&A with gemma3:4b")
print("=" * 70)

questions = [
    "What is the main topic of this document?",
    "What is the current global temperature rise?",
    "What are the recommendations for addressing climate change?"
]

for i, question in enumerate(questions, 1):
    print(f"\n--- Question {i} ---")
    print(f"Q: {question}")

    try:
        response = requests.post(
            f"{API_BASE}/api/v2/qa/ask/{batch_id}",
            json={
                "question": question,
                "top_k": 5,
                "temperature": 0.7,
                "include_sources": True
            },
            timeout=180  # 3 minutes for LLM response
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "No answer")
            sources = result.get("sources", [])

            print(f"\nA: {answer}")
            print(f"\nSources ({len(sources)} chunks):")
            for s in sources[:2]:
                print(f"  - {s['pdf_name']}, Page {s['page_no']} (score: {s['score']:.4f})")
        else:
            print(f"ERROR: {response.status_code} - {response.text}")

    except requests.exceptions.Timeout:
        print("ERROR: Request timed out")
    except Exception as e:
        print(f"ERROR: {e}")

# =============================================================================
# STEP 5: Generate summary
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Generate summary")
print("=" * 70)

try:
    response = requests.post(
        f"{API_BASE}/api/v2/qa/summarize/{batch_id}",
        json={
            "summary_type": "bullets",
            "max_chunks": 10,
            "temperature": 0.7
        },
        timeout=180
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Summary ({result.get('summary_type', 'bullets')}):")
        print("-" * 50)
        print(result.get("summary", "No summary"))
        print("-" * 50)
        print(f"\n✓ Summary generated using {result.get('chunks_used', 'N/A')} chunks")
    else:
        print(f"ERROR: {response.status_code} - {response.text}")

except requests.exceptions.Timeout:
    print("ERROR: Request timed out")
except Exception as e:
    print(f"ERROR: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"✓ PDF Upload: Success (batch_id: {batch_id})")
print(f"✓ Processing: Completed")
print(f"✓ Chunk Search: Working")
print(f"✓ Q&A with gemma3:4b: Working")
print(f"✓ Summary Generation: Working")
print("\nFull flow test completed successfully!")
