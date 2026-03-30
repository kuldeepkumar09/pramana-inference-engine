#!/usr/bin/env python
"""Test new API endpoints: explain, batch, debug."""

import json
import time
import sys
import requests
from threading import Thread
from pramana_engine.web import create_app

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Start Flask app in background thread
app = create_app()
server_thread = Thread(target=lambda: app.run(debug=False, use_reloader=False, port=5000), daemon=True)
server_thread.start()

# Wait for server to start
time.sleep(2)

BASE_URL = "http://localhost:5000/api/rag"

def safe_print(text: str):
    """Print text safely, replacing problematic Unicode."""
    safe_text = text.encode('ascii', 'replace').decode('ascii')
    print(safe_text)

def test_explain():
    """Test the /api/rag/explain endpoint."""
    print("\n" + "="*70)
    print("TEST: EXPLAIN ENDPOINT")
    print("="*70)
    
    question = "In a truth-seeking debate, what is the Nyaya debate type?"
    
    try:
        response = requests.post(
            f"{BASE_URL}/explain",
            json={"question": question, "useLLM": True},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        safe_print(f"\nQuestion: {question}")
        print(f"\nQuestion Type Detection:")
        for key, value in data.get("question_type", {}).items():
            print(f"  {key}: {value}")
        
        print(f"\nFallback Path:")
        for step in data.get("fallback_path", []):
            safe_print(f"  {step}")
        
        print(f"\nDecision Summary:")
        print(f"  Evidence Found: {data.get('evidence_found')}")
        print(f"  Clue Inferred: {data.get('clue_inferred')}")
        label = data.get('selected_label', 'N/A')
        safe_print(f"  Selected Label: {label}")
        reason = data.get('reason', '')
        safe_print(f"  Reason: {reason}")
        
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_batch():
    """Test the /api/rag/batch endpoint."""
    print("\n" + "="*70)
    print("TEST: BATCH ENDPOINT")
    print("="*70)
    
    questions = [
        "What are the five types of Hetvabhasa?",
        "Which debate fault is word twisting?",
        "What is a truth-seeking debate called?",
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch",
            json={"questions": questions, "useLLM": True},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"\nBatch Size: {data.get('batch_size')}")
        print(f"Succeeded: {data.get('succeeded')}/{data.get('batch_size')}")
        
        for i, result in enumerate(data.get("results", []), 1):
            q_preview = questions[i-1][:50] if i-1 < len(questions) else "unknown"
            print(f"\n  Question {i}: {q_preview}...")
            if "error" in result:
                print(f"    Status: ERROR - {result['error']}")
            else:
                answer_preview = result.get('answer', 'N/A')[:80]
                safe_print(f"    Answer: {answer_preview}")
                print(f"    Source: {result.get('answer_source', 'N/A')}")
        
        return data.get('succeeded', 0) == len(questions)
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_search():
    """Test the /api/rag/search endpoint (existing feature)."""
    print("\n" + "="*70)
    print("TEST: SEARCH ENDPOINT (Existing)")
    print("="*70)
    
    question = "What is Nyaya?"
    
    try:
        response = requests.post(
            f"{BASE_URL}/search",
            json={"question": question, "k": 5},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        safe_print(f"\nQuestion: {question}")
        print(f"Results Found: {data.get('count')}")
        
        for i, result in enumerate(data.get("results", [])[:3], 1):
            print(f"\n  Result {i}:")
            text_preview = result.get('text', 'N/A')[:80]
            safe_print(f"    Text: {text_preview}")
            print(f"    Score: {result.get('fused_score', 'N/A'):.3f}")
        
        return data.get('count', 0) > 0
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


# Run tests
print("\n" + "="*70)
print("NEW FEATURE TESTS")
print("="*70)

results = []
results.append(("EXPLAIN Endpoint", test_explain()))
results.append(("BATCH Endpoint", test_batch()))
results.append(("SEARCH Endpoint", test_search()))

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for name, passed in results:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status}: {name}")

passed_count = sum(1 for _, p in results if p)
print(f"\nTotal: {passed_count}/{len(results)} passed")

if passed_count == len(results):
    print("\n[SUCCESS] All new features are working!")
else:
    print(f"\n[WARNING] {len(results) - passed_count} feature(s) need investigation")
