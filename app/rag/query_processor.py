"""
Query processing: classification, expansion, and optimization
"""
import re
from typing import List, Tuple
import logging

from app.config import rag_config
from app.models.schemas import QueryType

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Process and classify user queries
    Handles query expansion and type detection
    """

    def __init__(self):
        self.patterns = rag_config.query_patterns

    def process(self, query: str) -> Tuple[str, QueryType, List[str]]:
        """
        Process query: classify type and optionally expand

        Args:
            query: Raw user query

        Returns:
            Tuple of (processed_query, query_type, expansion_terms)
        """
        # Classify query type
        query_type = self.classify_query(query)

        # Expand query if enabled
        expanded_terms = []
        if rag_config.query_expansion_enabled:
            expanded_terms = self.expand_query(query, query_type)

        # Clean query
        processed_query = self.clean_query(query)

        return processed_query, query_type, expanded_terms

    def classify_query(self, query: str) -> QueryType:
        """
        Classify query type based on patterns

        Args:
            query: User query

        Returns:
            QueryType enum
        """
        query_lower = query.lower()

        # Check each pattern category
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return QueryType(query_type)

        # Default to general
        return QueryType.GENERAL

    def expand_query(self, query: str, query_type: QueryType) -> List[str]:
        """
        Expand query with synonyms and related terms

        Args:
            query: Original query
            query_type: Classified query type

        Returns:
            List of expansion terms
        """
        expansions = []

        # Type-specific expansions
        if query_type == QueryType.CODE_SEARCH:
            # Add common code-related terms
            if 'controller' in query.lower():
                expansions.extend(['@RestController', '@RequestMapping', 'endpoint'])
            if 'service' in query.lower():
                expansions.extend(['@Service', 'business logic'])
            if 'repository' in query.lower():
                expansions.extend(['@Repository', 'database', 'query'])

        elif query_type == QueryType.SCHEMA:
            if 'table' in query.lower():
                expansions.extend(['CREATE TABLE', 'columns', 'schema'])
            if 'relationship' in query.lower():
                expansions.extend(['foreign key', 'references', 'join'])

        elif query_type == QueryType.DEBUGGING:
            if 'error' in query.lower():
                expansions.extend(['exception', 'stack trace', 'bug'])

        return expansions[:3]  # Limit to 3 expansions

    def clean_query(self, query: str) -> str:
        """
        Clean and normalize query

        Args:
            query: Raw query

        Returns:
            Cleaned query
        """
        # Remove extra whitespace
        query = ' '.join(query.split())

        # Remove special characters (keep alphanumeric and common punctuation)
        query = re.sub(r'[^\w\s\-\.\?\!]', '', query)

        return query.strip()


if __name__ == "__main__":
    processor = QueryProcessor()

    test_queries = [
        "How does UserController handle authentication?",
        "Explain the database schema for orders table",
        "Debug NullPointerException in OrderService",
        "What features does the system support?"
    ]

    for query in test_queries:
        processed, qtype, expansions = processor.process(query)
        print(f"Query: {query}")
        print(f"Type: {qtype}")
        print(f"Expansions: {expansions}")
        print()