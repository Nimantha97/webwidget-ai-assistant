
"""
Build RAG Pipeline - Complete Data Ingestion
Run this script to ingest all WebWidget data and build indexes
"""

import sys
import logging
from pathlib import Path
from app.rag.ingestion import KnowledgeBaseIngestion

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.ingestion import build_full_index
from app.config import get_data_path, LOGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'build_rag.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main build process"""
    logger.info("=" * 80)
    logger.info("WebWidget AI Chatbot - RAG Pipeline Build")
    logger.info("=" * 80)

    # ===================================================================
    # CONFIGURE YOUR WEBWIDGET PROJECT PATH HERE
    # ===================================================================
    WEBWIDGET_ROOT = r"C:\NewRepo\ideabiz-web-widgets"

    # Derived paths
    JAVA_SRC = Path(WEBWIDGET_ROOT) / "src" / "main" / "java"
    DOCS_PATH = Path(WEBWIDGET_ROOT) / "docs"

    logger.info(f"\n📂 WebWidget Project Location: {WEBWIDGET_ROOT}")
    logger.info(f"   Java Source: {JAVA_SRC}")
    logger.info(f"   Documentation: {DOCS_PATH}")

    # Check if paths exist
    if not JAVA_SRC.exists():
        logger.error(f"\n❌ ERROR: Java source path not found!")
        logger.error(f"   Expected: {JAVA_SRC}")
        logger.error(f"\n💡 Solution:")
        logger.error(f"   1. Check if the path is correct")
        logger.error(f"   2. Edit WEBWIDGET_ROOT in this script (line ~55)")
        logger.error(f"   3. Current path: {WEBWIDGET_ROOT}")
        return 1

    # Build index
    logger.info("\nStarting ingestion...")
    try:
        ingestion = KnowledgeBaseIngestion()

        # Ingest Java code from your actual project
        logger.info(f"\n📥 Ingesting Java code from your project...")
        ingestion.ingest_directory(str(JAVA_SRC), "*.java")

        # Ingest docs if they exist
        if DOCS_PATH.exists():
            logger.info(f"📥 Ingesting documentation...")
            ingestion.ingest_directory(str(DOCS_PATH), "*.md")
            ingestion.ingest_directory(str(DOCS_PATH), "*.txt")

        # Build vector index
        logger.info(f"\n🔨 Building vector index...")
        ingestion.build_vector_index()

        # Print statistics
        stats = ingestion.get_stats()
        logger.info("\n" + "=" * 80)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total chunks created: {stats['total_chunks']}")
        logger.info(f"Total indexed in vector DB: {stats['total_indexed']}")
        logger.info("\nDocument types:")
        for doc_type, count in stats['doc_types'].items():
            logger.info(f"  {doc_type}: {count} chunks")

        logger.info("\n✓ RAG pipeline successfully built!")
        logger.info("  You can now start the chatbot:")
        logger.info("  $ streamlit run ui/streamlit_app.py")
        logger.info("  or")
        logger.info("  $ uvicorn main:app --reload")

        return 0

    except Exception as e:
        logger.error(f"\n✗ Build failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())