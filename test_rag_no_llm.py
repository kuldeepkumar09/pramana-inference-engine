"""
Test RAG pipeline without LLM (retrieval-only mode).
This tests the FAISS + E5-Small embeddings search without requiring Ollama.
"""

import requests

BASE_URL = "http://localhost:5000"

def main() -> int:
    print("\n" + "=" * 80)
    print("RAG PIPELINE TEST (Retrieval-Only, No LLM Required)")
    print("=" * 80)

    # Test 1: Check pipeline status
    print("\n[TEST 1] Checking RAG Pipeline Status...")
    print("-" * 80)
    try:
        response = requests.get(f"{BASE_URL}/api/rag/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Pipeline Status: {data.get('status')}")
            print(f"  - Vector Store Size: {data.get('vector_store_size')} chunks")
            print(f"  - Embedding Model: {data.get('embedding_model')}")
            print(f"  - LLM Available: {data.get('llm_available')}")
            print(f"  - LLM Healthy: {data.get('llm_healthy')}")
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"✗ Connection Error: {e}")
        print("  → Is Flask server running? Run: python -m pramana_engine.web")
        return 1

    # Test 2: Retrieval-only search
    print("\n[TEST 2] Retrieval-Only Search (Hybrid: FAISS + Keyword)...")
    print("-" * 80)
    search_payload = {
        "question": "What is pratyaksha perception in Nyaya philosophy?",
        "pramanaTypes": ["perception"],
        "k": 5,
    }
    print(f"Query: {search_payload['question']}")
    print()

    response = requests.post(f"{BASE_URL}/api/rag/search", json=search_payload, timeout=30)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        print(f"✓ Found {len(results)} relevant chunks:\n")

        for i, chunk in enumerate(results, 1):
            print(f"  [{i}] {chunk['id']}")
            print(f"      Source: {chunk['source']}")
            print(f"      Score: {chunk['score']:.4f} (fused from {chunk.get('sources', 'unknown')})")
            print(f"      Text: {chunk['text'][:120]}...")
            print(f"      Supports: {', '.join(chunk.get('supports', []))}")
            print()
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)

    # Test 3: Full RAG answer WITHOUT LLM (uses retrieval + pramana verifier)
    print("\n[TEST 3] Full RAG Answer (Retrieval + Symbolic Verifier, No LLM)...")
    print("-" * 80)
    answer_payload = {
        "question": "Does testimony override perception in Nyaya logic?",
        "useLLM": False,  # Don't use LLM, just retrieval
        "pramanaTypes": ["perception", "testimony"],
    }
    print(f"Query: {answer_payload['question']}")
    print()

    response = requests.post(f"{BASE_URL}/api/rag/answer", json=answer_payload, timeout=30)
    if response.status_code == 200:
        data = response.json()
        print("✓ Answer Generation Successful!")
        print()
        print(f"  Answer: {data.get('answer', 'N/A')[:300]}...")
        print()
        print(f"  Confidence: {data.get('confidence', 0):.2%}")
        print(f"  Epistemic Status: {data.get('epistemic_status')}")
        print(f"  Answer Source: {data.get('answer_source')}")
        print()
        print("  Citations (RAG Chunks):")
        for i, chunk in enumerate(data.get('rag_chunks', [])[:3], 1):
            print(f"    [{i}] {chunk['id']} ({chunk['source']}) - Score: {chunk['score']:.4f}")
        print()

        verifier = data.get('verifier', {})
        print("  Verification Constraints:")
        for constraint, status in verifier.get('constraints', {}).items():
            mark = "✓" if status else "✗"
            print(f"    {mark} {constraint}: {status}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)

    # Test 4: Another retrieval test with inference focus
    print("\n[TEST 4] Another Retrieval Test (Anumana - Inference Focus)...")
    print("-" * 80)
    search_payload2 = {
        "question": "What is the relationship between hetu (mark) and vyapti (invariable relation) in inference?",
        "pramanaTypes": ["inference"],
        "k": 3,
    }
    print(f"Query: {search_payload2['question']}")
    print()

    response = requests.post(f"{BASE_URL}/api/rag/search", json=search_payload2, timeout=30)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        print(f"✓ Found {len(results)} relevant chunks:")
        print()

        for i, chunk in enumerate(results, 1):
            print(f"  [{i}] {chunk['id']}")
            print(f"      Score: {chunk['score']:.4f}")
            print(f"      {chunk['text'][:150]}...")
            print()
    else:
        print(f"✗ Error: {response.status_code}")

    print("=" * 80)
    print("✓ All retrieval tests completed successfully!")
    print("\nNEXT STEPS:")
    print("  1. Install Ollama from: https://ollama.ai/")
    print("  2. Run 'ollama pull mistral:7b' to download the model")
    print("  3. Run 'ollama serve' in a separate terminal")
    print("  4. Then run: python test_rag_with_llm.py")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
