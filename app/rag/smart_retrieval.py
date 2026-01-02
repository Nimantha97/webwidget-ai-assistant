"""
Smart Retrieval Wrapper
Adapts top_k based on query intent
"""
import logging
import re
from typing import List, Optional
from app.rag.retrieval import HybridRetriever
from app.models.schemas import QueryType, RetrievedChunk

logger = logging.getLogger(__name__)


class SmartRetriever:
    """
    Wrapper around HybridRetriever that adapts top_k based on query

    Use cases:
    - "List all controllers" → top_k=20 (get many results)
    - "What does UserController do?" → top_k=3 (get focused results)
    """

    def __init__(self):
        self.base_retriever = HybridRetriever()

        # Patterns that indicate "list all" intent
        self.list_patterns = [
            r'\ball\b',
            r'\blist\b',
            r'\bshow (?:me )?all\b',
            r'\bevery\b',
            r'\bcomplete list\b',
            r'\bfull list\b',
        ]

        # Patterns for specific searches
        self.specific_patterns = [
            r'\bwhat (?:does|is)\b',
            r'\bhow (?:does|to)\b',
            r'\bexplain\b',
            r'\bwhy\b',
        ]

    def retrieve(
            self,
            query: str,
            query_type: Optional[QueryType] = None,
            top_k: Optional[int] = None,
            use_graph: bool = False
    ) -> List[RetrievedChunk]:
        """
        Smart retrieval with adaptive top_k

        Args:
            query: User's question
            query_type: Type of query
            top_k: Override (if None, will auto-detect)
            use_graph: Whether to use graph

        Returns:
            Retrieved chunks
        """
        # Auto-detect top_k if not provided
        if top_k is None:
            top_k = self._detect_optimal_top_k(query)

        logger.info(f"🎯 Adaptive top_k: {top_k} (query: '{query[:50]}...')")

        # Call base retriever
        return self.base_retriever.retrieve(
            query=query,
            query_type=query_type,
            top_k=top_k,
            use_graph=use_graph
        )

    def _detect_optimal_top_k(self, query: str) -> int:
        """
        Detect optimal top_k based on query intent

        Returns:
            top_k value (3-20)
        """
        query_lower = query.lower()

        # Check for "list all" patterns
        for pattern in self.list_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"   Detected 'list all' intent → top_k=15")
                return 15  # Get more results for listing

        # Check for specific search patterns
        for pattern in self.specific_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"   Detected specific search → top_k=3")
                return 3  # Focused results

        # Default: moderate
        logger.info(f"   Default query → top_k=5")
        return 5


# Singleton
_smart_retriever = None


def get_smart_retriever() -> SmartRetriever:
    """Get or create SmartRetriever singleton"""
    global _smart_retriever

    if _smart_retriever is None:
        _smart_retriever = SmartRetriever()

    return _smart_retriever


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    retriever = SmartRetriever()

    # Test cases
    test_queries = [
        ("List all controllers in the project", QueryType.CODE_SEARCH),
        ("What does UserController do?", QueryType.CODE_SEARCH),
        ("Show me every service class", QueryType.CODE_SEARCH),
        ("How does authentication work?", QueryType.DOCUMENTATION),
    ]

    for query, qtype in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")

        results = retriever.retrieve(query, qtype)

        print(f"Retrieved: {len(results)} chunks")
        for i, chunk in enumerate(results[:5], 1):
            print(f"  {i}. {chunk.metadata.get('filename', 'unknown')}")