"""
Test RAG pipeline WITH LLM (Mistral 7B via Ollama).
Requires: Ollama running with mistral:7b downloaded.
"""

import requests
import time

BASE_URL = "http://localhost:5000"

def main() -> int:
    print("\n" + "=" * 80)
    print("RAG PIPELINE TEST (With Mistral 7B LLM)")
    print("=" * 80)
    print("\nIMPORTANT: Ensure Ollama is running!")
    print("  In another terminal, run: ollama serve")
    print()

    # Check if Ollama is available
    print("[SETUP] Checking Ollama availability...")
    response = requests.get(f"{BASE_URL}/api/rag/status", timeout=10)
    if response.status_code == 200:
        data = response.json()
        if not data.get('llm_healthy'):
            print("✗ ERROR: Ollama server is not running or Mistral 7B not downloaded")
            print("\nTo fix:")
            print("  1. Start Ollama: ollama serve")
            print("  2. Download model: ollama pull mistral:7b")
            print("  3. Try again")
            return 1
        print("✓ Ollama is healthy!")
        print(f"  - LLM Model: {data.get('llm_model')}")
        print(f"  - Vector Store: {data.get('vector_store_size')} chunks")
    else:
        print("✗ Cannot connect to Flask server")
        return 1

    # Test 1: Simple LLM reasoning (no chain-of-thought)
    print("\n" + "=" * 80)
    print("[TEST 1] Simple LLM Reasoning (Fast)")
    print("=" * 80)
    payload1 = {
        "question": "What is pratyaksha perception according to Nyaya?",
        "useLLM": True,
        "useReasoningChain": False,  # Faster, simpler reasoning
    }
    print(f"Query: {payload1['question']}")
    print("\nGenerating answer with Mistral 7B... (this may take 30-60 seconds)")
    print()

    start = time.time()
    response = requests.post(f"{BASE_URL}/api/rag/answer", json=payload1, timeout=120)
    elapsed = time.time() - start

    if response.status_code == 200:
        data = response.json()
        print(f"✓ LLM Answer Generated in {elapsed:.1f}s")
        print()
        print(f"  Question: {data.get('question')}")
        print(f"  Answer Source: {data.get('answer_source')}")
        print()
        print("  LLM Response:")
        print(f"  {data.get('answer', 'N/A')}")
        print()
        print(f"  Confidence: {data.get('confidence', 0):.2%}")
        print(f"  Epistemic Status: {data.get('epistemic_status')}")
        print()
        print(f"  Top Citation: {data['rag_chunks'][0]['source'] if data.get('rag_chunks') else 'N/A'}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text[:500])

    # Test 2: Chain-of-Thought reasoning (detailed, slower)
    print("\n" + "=" * 80)
    print("[TEST 2] Chain-of-Thought LLM Reasoning (Detailed)")
    print("=" * 80)
    payload2 = {
        "question": "In Nyaya, what is the relationship between hetu (inferential mark) and vyapti (invariable relation)?",
        "useLLM": True,
        "useReasoningChain": True,  # Detailed reasoning steps
    }
    print(f"Query: {payload2['question']}")
    print("\nGenerating detailed reasoning with Mistral 7B... (this may take 60-120 seconds)")
    print()

    start = time.time()
    response = requests.post(f"{BASE_URL}/api/rag/answer", json=payload2, timeout=180)
    elapsed = time.time() - start

    if response.status_code == 200:
        data = response.json()
        print(f"✓ CoT Reasoning Generated in {elapsed:.1f}s")
        print()

        cot = data.get('cot_reasoning', {})
        print("  Understanding:")
        print(f"  {cot.get('understanding', 'N/A')}")
        print()
        print("  Relevant Evidence:")
        print(f"  {cot.get('evidence', 'N/A')}")
        print()
        print("  Reasoning Steps:")
        print(f"  {cot.get('steps', 'N/A')}")
        print()
        print("  Final Answer:")
        print(f"  {data.get('answer', 'N/A')}")
        print()
        print(f"  Confidence: {data.get('confidence', 0):.2%}")
        print(f"  LLM Confidence: {cot.get('llm_confidence', 0):.2%}")
        print(f"  Epistemic Status: {data.get('epistemic_status')}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text[:500])

    # Test 3: Temperature/Creativity variation
    print("\n" + "=" * 80)
    print("[TEST 3] Comparative Queries")
    print("=" * 80)

    questions = [
        "Does testimony ever contradict perception in Nyaya?",
        "What are the four pramanas?",
        "How does arthapatti (postulation) work in inference?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n  Query {i}: {q}")
        payload = {"question": q, "useLLM": True, "useReasoningChain": False}

        response = requests.post(f"{BASE_URL}/api/rag/answer", json=payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            answer_preview = data.get('answer', 'N/A')[:100]
            conf = data.get('confidence', 0)
            status = data.get('epistemic_status', 'unknown')
            print(f"    ✓ {answer_preview}...")
            print(f"      Confidence: {conf:.2%}, Status: {status}")
        else:
            print(f"    ✗ Error: {response.status_code}")

    print("\n" + "=" * 80)
    print("✓ All LLM tests completed!")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
