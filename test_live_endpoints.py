#!/usr/bin/env python
"""Test live RAG endpoints with real Nyaya queries."""

import json
import sys
import time
import requests
from threading import Thread
from pramana_engine.web import create_app

# Start Flask app in background thread
app = create_app()
server_thread = Thread(target=lambda: app.run(debug=False, use_reloader=False, port=5000), daemon=True)
server_thread.start()

# Wait for server to start
time.sleep(2)

BASE_URL = "http://localhost:5000/api/rag"

def test_query(name: str, question: str, use_llm: bool = True):
    """Test a single query and print results."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Question: {question}\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/answer",
            json={
                "question": question,
                "useLLM": use_llm,
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"[OK] Answer: {data.get('answer', 'N/A')}")
        print(f"[OK] Source: {data.get('answer_source', 'N/A')}")
        print(f"[OK] Confidence: {data.get('confidence', 'N/A'):.2f}")
        print(f"[OK] Status: {data.get('epistemic_status', 'N/A')}")
        
        # Show first citation if available
        chunks = data.get('rag_chunks', [])
        if chunks:
            print(f"[OK] Top Citation: {chunks[0].get('excerpt', 'N/A')[:100]}...")
            
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


# Test cases covering all three Nyaya question types
tests = [
    ("HETVABHASA - INFERENCE", 
     "The reason proves both presence and absence. Which Hetvabhasa is this?"),
    
    ("HETVABHASA - CATALOG",
     "What are the five major types of Hetvabhasa in Nyaya?"),
    
    ("DEBATE FAULT - WITH CLUE",
     "Which Nyaya debate fault applies when someone intentionally twists the opponent's words?"),
    
    ("DEBATE FAULT - PURE REFUTATION",
     "In debate, this is pure refutation with no own thesis. Which Nyaya defect is this?"),
    
    ("DEBATE MODE - TRUTH-SEEKING (NEW)",
     "In a truth-seeking debate, what is the Nyaya debate type?"),
    
    ("DEBATE MODE - WINNING ORIENTED (NEW)",
     "What is the Nyaya debate mode aimed at winning an argument at any cost?"),
]

results = []
for name, question in tests:
    passed = test_query(name, question)
    results.append((name, passed))

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
passed_count = sum(1 for _, p in results if p)
for name, passed in results:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status}: {name}")

print(f"\nTotal: {passed_count}/{len(results)} passed")
sys.exit(0 if passed_count == len(results) else 1)
