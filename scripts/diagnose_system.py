"""
Complete System Diagnostic Script
Run this to verify your RAG pipeline is working correctly
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import chromadb
from chromadb.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_chromadb():
    """Check if ChromaDB is populated"""
    print("\n" + "=" * 80)
    print("1. CHECKING CHROMADB")
    print("=" * 80)

    try:
        from app.config import rag_config

        client = chromadb.PersistentClient(
            path=rag_config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        collection = client.get_collection(name=rag_config.chroma_collection_name)
        count = collection.count()

        print(f"✅ ChromaDB connected")
        print(f"   Location: {rag_config.chroma_persist_dir}")
        print(f"   Collection: {rag_config.chroma_collection_name}")
        print(f"   Documents: {count}")

        if count == 0:
            print("\n❌ CRITICAL: ChromaDB is EMPTY!")
            print("   FIX: Run 'python scripts/build_rag.py'")
            return False

        # Sample a few documents
        sample = collection.get(limit=3, include=["documents", "metadatas"])
        print(f"\n📄 Sample documents:")
        for i, (doc, meta) in enumerate(zip(sample['documents'], sample['metadatas']), 1):
            filename = meta.get('filename', 'unknown')
            print(f"   {i}. {filename} ({len(doc)} chars)")

        return True

    except Exception as e:
        print(f"❌ ChromaDB check failed: {e}")
        return False


def check_embedding_model():
    """Check if embedding model loads"""
    print("\n" + "=" * 80)
    print("2. CHECKING EMBEDDING MODEL")
    print("=" * 80)

    try:
        from sentence_transformers import SentenceTransformer
        from app.config import rag_config

        model = SentenceTransformer(rag_config.embedding_model, device='cpu')
        print(f"✅ Embedding model loaded")
        print(f"   Model: {rag_config.embedding_model}")
        print(f"   Dimension: {rag_config.embedding_dimension}")

        # Test encoding
        test_text = "This is a test"
        embedding = model.encode(test_text)
        print(f"   Test encoding: {len(embedding)} dimensions")

        return True

    except Exception as e:
        print(f"❌ Embedding model check failed: {e}")
        return False


def check_reranker():
    """Check if reranker loads (optional)"""
    print("\n" + "=" * 80)
    print("3. CHECKING RERANKER (Optional)")
    print("=" * 80)

    try:
        from sentence_transformers import CrossEncoder
        from app.config import rag_config

        if not rag_config.reranking_enabled:
            print("⏭️  Reranking disabled in config")
            return True

        reranker = CrossEncoder(
            rag_config.reranking_model,
            max_length=512,
            device='cpu',
            trust_remote_code=True
        )
        print(f"✅ Reranker loaded")
        print(f"   Model: {rag_config.reranking_model}")
        print(f"   This will boost accuracy by ~20%")

        return True

    except Exception as e:
        print(f"⚠️ Reranker failed to load: {e}")
        print(f"   System will work without it (lower accuracy)")
        return True  # Non-critical


def check_llm_model():
    """Check if LLM model file exists"""
    print("\n" + "=" * 80)
    print("4. CHECKING LLM MODEL")
    print("=" * 80)

    model_path = Path("data/models/qwen-2.5-coder-7b/qwen2.5-coder-7b-instruct-q4_k_m.gguf")

    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024 ** 3)
        print(f"✅ LLM model found")
        print(f"   Location: {model_path}")
        print(f"   Size: {size_gb:.2f} GB")
        return True
    else:
        print(f"❌ LLM model NOT FOUND")
        print(f"   Expected: {model_path}")
        print(f"   FIX: Download the model and place it here")
        return False


def test_retrieval():
    """Test actual retrieval"""
    print("\n" + "=" * 80)
    print("5. TESTING RETRIEVAL PIPELINE")
    print("=" * 80)

    try:
        from app.rag.retrieval import HybridRetriever
        from app.models.schemas import QueryType

        retriever = HybridRetriever()

        # Test queries
        test_cases = [
            ("UserController", QueryType.CODE_SEARCH),
            ("database table", QueryType.SCHEMA),
            ("how authentication works", QueryType.DOCUMENTATION),
        ]

        all_passed = True

        for query, qtype in test_cases:
            print(f"\n📝 Testing: '{query}' (type: {qtype.value})")

            results = retriever.retrieve(query, query_type=qtype, top_k=3)

            if results:
                print(f"   ✅ Retrieved {len(results)} chunks")
                print(f"   📊 Scores: {[f'{c.score:.3f}' for c in results]}")
                print(f"   📁 Sources: {[c.metadata.get('filename', 'unknown')[:30] for c in results]}")
            else:
                print(f"   ❌ No results retrieved!")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generator():
    """Test LLM generation (if model exists)"""
    print("\n" + "=" * 80)
    print("6. TESTING LLM GENERATION")
    print("=" * 80)

    model_path = Path("data/models/qwen-2.5-coder-7b/qwen2.5-coder-7b-instruct-q4_k_m.gguf")

    if not model_path.exists():
        print("⏭️  LLM model not found, skipping generation test")
        return True

    try:
        from app.llm.generator import get_generator
        from app.models.schemas import RetrievedChunk, QueryType

        print("⏳ Loading LLM (this may take 30 seconds)...")
        generator = get_generator()
        print("✅ LLM loaded")

        # Test with mock chunk
        test_chunk = RetrievedChunk(
            content="@RestController\npublic class UserController {\n  @GetMapping('/users')\n  public List<User> getUsers() {...}\n}",
            score=0.85,
            source="UserController.java",
            metadata={"filename": "UserController.java", "class_name": "UserController"},
            chunk_id="test_1"
        )

        print("\n⏳ Generating response (10-30 seconds)...")
        response = generator.generate(
            query="What does UserController do?",
            retrieved_chunks=[test_chunk],
            query_type=QueryType.CODE_SEARCH,
            max_tokens=200
        )

        print(f"\n✅ Generation successful!")
        print(f"📝 Response preview:")
        print(f"   {response[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostics"""
    print("\n" + "=" * 80)
    print("WEBWIDGET AI CHATBOT - SYSTEM DIAGNOSTIC")
    print("=" * 80)

    results = {
        "ChromaDB": check_chromadb(),
        "Embedding Model": check_embedding_model(),
        "Reranker": check_reranker(),
        "LLM Model": check_llm_model(),
        "Retrieval": test_retrieval(),
        "Generation": test_generator(),
    }

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {component}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL CHECKS PASSED - SYSTEM READY!")
        print("\nYou can now run:")
        print("  streamlit run ui/streamlit_app.py")
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES ABOVE")
        print("\nCommon fixes:")
        print("  1. ChromaDB empty → Run 'python scripts/build_rag.py'")
        print("  2. LLM model missing → Download and place in data/models/")
        print("  3. Retrieval fails → Check logs above for specific errors")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())