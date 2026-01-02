"""
Quick Test Script - Verify retrieval works with 0.001 threshold
Run this after applying the fix
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO)

from app.rag.retrieval import HybridRetriever
from app.models.schemas import QueryType

print("\n" + "=" * 80)
print("QUICK TEST - Verifying retrieval with 0.001 threshold")
print("=" * 80 + "\n")

try:
    # Initialize
    print("Initializing retriever...")
    retriever = HybridRetriever()

    # Test cases that should NOW work
    test_cases = [
        ("UserController", QueryType.CODE_SEARCH),
        ("database", QueryType.SCHEMA),
        ("configuration", QueryType.DOCUMENTATION),
    ]

    all_passed = True

    for query, qtype in test_cases:
        print(f"\n{'=' * 80}")
        print(f"TEST: '{query}' (type: {qtype.value})")
        print(f"{'=' * 80}\n")

        results = retriever.retrieve(query, query_type=qtype, top_k=3)

        if results:
            print(f"✅ SUCCESS: Retrieved {len(results)} chunks")
            for i, chunk in enumerate(results, 1):
                filename = chunk.metadata.get('filename', 'unknown')
                print(f"   {i}. {filename} (score: {chunk.score:.6f})")
        else:
            print(f"❌ FAILED: No results")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYour retrieval is now working correctly!")
        print("Next step: Run 'streamlit run ui/streamlit_app.py'")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nIf tests still fail:")
        print("1. Verify you replaced retrieval.py with the new version")
        print("2. Check that threshold is 0.001 (not 0.05)")
        print("3. Run 'python -m scripts.diagnose_system' for details")
    print("=" * 80 + "\n")

except Exception as e:
    print(f"\n❌ TEST FAILED WITH ERROR: {e}")
    import traceback

    traceback.print_exc()
    print("\nTroubleshooting:")
