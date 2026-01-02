"""
COMPLETE FIX SCRIPT - Diagnose and Fix Accuracy Issues
Run this to identify and fix all problems
"""
import sys
import logging
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def diagnose_chunking():
    """Diagnose chunking quality"""
    print_section("1. DIAGNOSING CHUNKING QUALITY")

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path="data/chroma",
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(name="webwidget_knowledge")

        # Get sample chunks
        results = collection.get(limit=10, include=["documents", "metadatas"])

        chunk_sizes = [len(doc) for doc in results['documents']]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)

        print(f"✓ Total chunks: {collection.count()}")
        print(f"✓ Average chunk size: {avg_size:.0f} characters")
        print(f"✓ Min size: {min(chunk_sizes)}")
        print(f"✓ Max size: {max(chunk_sizes)}")

        # Check for overly large chunks
        large_chunks = [s for s in chunk_sizes if s > 2000]
        if large_chunks:
            print(f"⚠️  WARNING: {len(large_chunks)} chunks are > 2000 chars (too large!)")
            print("   Solution: Reduce chunk_size in config.yaml")
            return False

        # Check for too many chunks
        if collection.count() > 2000:
            print(f"⚠️  WARNING: {collection.count()} chunks may be too many")
            print("   Solution: Increase chunk_size slightly")

        print("✅ Chunking looks good")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def diagnose_retrieval():
    """Diagnose retrieval quality"""
    print_section("2. DIAGNOSING RETRIEVAL QUALITY")

    try:
        from app.rag.retrieval import HybridRetriever
        from app.rag.query_processor import QueryProcessor
        from app.models.schemas import QueryType

        retriever = HybridRetriever()
        processor = QueryProcessor()

        # Test queries
        test_queries = [
            ("UserController authentication", QueryType.CODE_SEARCH),
            ("database schema", QueryType.SCHEMA),
            ("how does login work", QueryType.DEBUGGING)
        ]

        print("\nTesting retrieval accuracy...")

        all_good = True
        for query, qtype in test_queries:
            print(f"\n📝 Query: '{query}' (type: {qtype.value})")

            processed_query, _, _ = processor.process(query)
            chunks = retriever.retrieve(
                query=processed_query,
                query_type=qtype,
                top_k=5
            )

            if not chunks:
                print(f"   ❌ NO RESULTS - This is bad!")
                all_good = False
                continue

            print(f"   ✓ Retrieved {len(chunks)} chunks")
            print(f"   Top scores: {[f'{c.score:.3f}' for c in chunks[:3]]}")

            # Check relevance scores
            if chunks[0].score < 0.5:
                print(f"   ⚠️  Low relevance score ({chunks[0].score:.3f})")
                print("   Solution: Enable reranking or rebuild index")
                all_good = False
            else:
                print(f"   ✅ Good relevance ({chunks[0].score:.3f})")

        if all_good:
            print("\n✅ Retrieval quality is good")
        else:
            print("\n⚠️  Retrieval needs improvement")

        return all_good

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def diagnose_llm():
    """Diagnose LLM setup"""
    print_section("3. DIAGNOSING LLM CONFIGURATION")

    try:
        from app.llm.generator import get_generator
        from app.models.schemas import RetrievedChunk, QueryType

        print("Loading LLM (30 seconds)...")
        start = time.time()
        generator = get_generator()
        load_time = time.time() - start

        print(f"✓ LLM loaded in {load_time:.1f} seconds")

        if load_time > 45:
            print("⚠️  Load time is slow (>45s)")

        # Check context window
        if hasattr(generator.llm, 'n_ctx'):
            ctx = generator.llm.n_ctx()
            print(f"✓ Context window: {ctx} tokens")

            if ctx < 8192:
                print(f"⚠️  Context window is small ({ctx} tokens)")
                print("   Solution: Increase n_ctx in generator.py to 32768")
                return False

        # Test generation
        print("\nTesting generation speed...")
        test_chunks = [
            RetrievedChunk(
                content="@RestController\npublic class UserController {\n  @GetMapping('/users')\n  public List<User> getUsers() {...}\n}",
                score=0.95,
                source="UserController.java",
                metadata={"class_name": "UserController"},
                chunk_id="test_1"
            )
        ]

        start = time.time()
        response = generator.generate(
            query="What does UserController do?",
            retrieved_chunks=test_chunks,
            query_type=QueryType.CODE_SEARCH,
            max_tokens=200
        )
        gen_time = time.time() - start

        print(f"✓ Generation time: {gen_time:.1f} seconds")
        print(f"✓ Response length: {len(response)} characters")

        if gen_time > 30:
            print("⚠️  Generation is VERY slow (>30s for 200 tokens)")
            print("   Possible causes:")
            print("   - Context window too small (causing overflow)")
            print("   - Too many chunks sent to LLM")
            print("   - CPU is slow")
            return False
        elif gen_time > 15:
            print("⚠️  Generation is slow but acceptable")
        else:
            print("✅ Generation speed is good")

        # Check response quality
        if len(response) < 50:
            print("⚠️  Response is very short - may indicate problem")
            return False

        print("\n✅ LLM is working correctly")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def diagnose_context_overflow():
    """Diagnose context overflow issues"""
    print_section("4. DIAGNOSING CONTEXT OVERFLOW")

    try:
        from app.rag.retrieval import HybridRetriever
        from app.llm.prompts import format_context_optimized
        from app.models.schemas import QueryType

        retriever = HybridRetriever()

        # Test with complex query
        query = "Explain the complete user authentication and authorization flow in WebWidget"
        chunks = retriever.retrieve(query, query_type=QueryType.CODE_SEARCH, top_k=10)

        print(f"Retrieved {len(chunks)} chunks")

        # Check total content size
        total_chars = sum(len(c.content) for c in chunks)
        print(f"Total content: {total_chars:,} characters")
        print(f"Estimated tokens: {total_chars // 4:,}")

        if total_chars > 16000:
            print("❌ CRITICAL: Context is TOO LARGE!")
            print(f"   Current: {total_chars:,} chars ({total_chars // 4:,} tokens)")
            print(f"   Maximum safe: 16,000 chars (4,000 tokens)")
            print("\n   SOLUTIONS:")
            print("   1. Reduce top_k in retrieval (use 3-5 max)")
            print("   2. Use format_context_optimized() with max_length=4000")
            print("   3. Enable aggressive chunk filtering")
            return False

        # Test optimized formatting
        formatted = format_context_optimized(chunks, max_length=4000, max_chunks=3)
        formatted_chars = len(formatted)

        print(f"\nAfter optimization:")
        print(f"  Chunks used: 3")
        print(f"  Context size: {formatted_chars:,} characters")
        print(f"  Estimated tokens: {formatted_chars // 4:,}")

        if formatted_chars > 6000:
            print("⚠️  Still large after optimization")
            print("   Solution: Rebuild with smaller chunk_size")
            return False

        print("✅ Context size is manageable")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def provide_fixes(issues):
    """Provide specific fixes based on issues found"""
    print_section("RECOMMENDED FIXES")

    if not any(issues):
        print("\n🎉 NO MAJOR ISSUES FOUND!")
        print("\nYour system should be working well. If you still have problems:")
        print("1. Check that reranking is enabled (config.yaml)")
        print("2. Reduce max_tokens to 1024")
        print("3. Use only top 3 chunks (set top_k=3)")
        return

    print("\n⚠️  ISSUES FOUND - HERE'S HOW TO FIX:")

    if not issues[0]:  # Chunking
        print("\n📌 CHUNKING ISSUES:")
        print("   1. Edit config.yaml:")
        print("      chunking.code.chunk_size: 800 (was 1000)")
        print("      chunking.markdown.chunk_size: 600 (was 800)")
        print("   2. Rebuild index: python scripts/build_rag.py")

    if not issues[1]:  # Retrieval
        print("\n📌 RETRIEVAL ISSUES:")
        print("   1. Edit config.yaml:")
        print("      reranking.enabled: true")
        print("      reranking.top_k: 3")
        print("   2. Restart application")

    if not issues[2]:  # LLM
        print("\n📌 LLM ISSUES:")
        print("   1. Edit app/llm/generator.py line 37:")
        print("      n_ctx=32768 (was 4096)")
        print("   2. Reduce max_tokens to 1024")
        print("   3. Use only 3 chunks maximum")

    if not issues[3]:  # Context overflow
        print("\n📌 CONTEXT OVERFLOW:")
        print("   1. In retrieval.retrieve(), set top_k=3 (not 5 or 10)")
        print("   2. Use format_context_optimized() with max_length=4000")
        print("   3. Enable chunk filtering (min_score=0.5)")

    print("\n" + "=" * 80)
    print("QUICK FIX SUMMARY:")
    print("=" * 80)
    print("""
1. Replace files with OPTIMIZED versions from artifacts:
   - app/llm/generator.py (with n_ctx=32768)
   - app/llm/prompts.py (with format_context_optimized)
   - app/rag/retrieval.py (with better filtering)
   - config.yaml (with optimized settings)

2. Rebuild index with smaller chunks:
   python scripts/build_rag.py

3. Restart application:
   streamlit run ui/streamlit_app.py

4. Test with simple query:
   "What does UserController do?"
   Should respond in 10-20 seconds
""")


def main():
    """Run complete diagnosis"""
    print_section("WEBWIDGET AI CHATBOT - ACCURACY & PERFORMANCE DIAGNOSIS")

    print("""
This script will:
1. Check chunking quality
2. Test retrieval accuracy
3. Diagnose LLM configuration
4. Check for context overflow issues
5. Provide specific fixes

Starting diagnosis...
""")

    issues = [
        diagnose_chunking(),
        diagnose_retrieval(),
        diagnose_llm(),
        diagnose_context_overflow()
    ]

    provide_fixes(issues)

    print("\n" + "=" * 80)
    if all(issues):
        print("✅ ALL CHECKS PASSED - Your system is well configured!")
    else:
        print("⚠️  ISSUES FOUND - Follow the fixes above")
    print("=" * 80)


if __name__ == "__main__":
    main()