"""
Multi-modal document chunkers for different data types
Specialized strategies for documentation, code, and schemas
"""
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)
from langchain_core.documents import Document
import javalang

from app.config import rag_config


class BaseChunker:
    """Base class for chunkers"""

    def __init__(self):
        self.chunks = []

    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Chunk content and return list of Documents"""
        raise NotImplementedError


class MarkdownChunker(BaseChunker):
    """
    Hierarchical header-based chunking for Markdown documentation
    Preserves document structure and headers for context
    """

    def __init__(self):
        super().__init__()
        # Define headers to split on
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

        # Header-based splitter
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False  # Keep headers for context
        )

        # Secondary recursive splitter for large sections
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag_config.chunk_size_markdown,
            chunk_overlap=rag_config.chunk_overlap_markdown,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Chunk markdown content preserving header hierarchy

        Args:
            content: Markdown text
            metadata: File metadata

        Returns:
            List of Document objects with enriched metadata
        """
        # First split by headers
        header_splits = self.header_splitter.split_text(content)

        # Then split large sections
        documents = []
        for i, doc in enumerate(header_splits):
            # Extract header path from metadata
            header_path = []
            for j in range(1, 5):
                header_key = f"Header {j}"
                if header_key in doc.metadata:
                    header_path.append(doc.metadata[header_key])

            # Check if chunk is too large
            if len(doc.page_content) > rag_config.chunk_size_markdown * 1.5:
                # Further split large sections
                sub_chunks = self.text_splitter.split_text(doc.page_content)
                for j, chunk_text in enumerate(sub_chunks):
                    chunk_metadata = {
                        **metadata,
                        **doc.metadata,
                        "chunk_id": f"{metadata.get('source', 'unknown')}_chunk_{i}_{j}",
                        "doc_type": "documentation",
                        "header_path": header_path,
                        "chunk_position": i * 100 + j,
                        "has_code_block": "```" in chunk_text,
                        "has_table": "|" in chunk_text and "---" in chunk_text,
                    }
                    # Add contextual prefix
                    context_prefix = self._create_context_prefix(header_path)
                    enriched_content = f"{context_prefix}\n\n{chunk_text}"

                    documents.append(Document(
                        page_content=enriched_content,
                        metadata=chunk_metadata
                    ))
            else:
                # Keep as single chunk
                chunk_metadata = {
                    **metadata,
                    **doc.metadata,
                    "chunk_id": f"{metadata.get('source', 'unknown')}_chunk_{i}",
                    "doc_type": "documentation",
                    "header_path": header_path,
                    "chunk_position": i,
                    "has_code_block": "```" in doc.page_content,
                    "has_table": "|" in doc.page_content and "---" in doc.page_content,
                }
                context_prefix = self._create_context_prefix(header_path)
                enriched_content = f"{context_prefix}\n\n{doc.page_content}"

                documents.append(Document(
                    page_content=enriched_content,
                    metadata=chunk_metadata
                ))

        return documents

    def _create_context_prefix(self, header_path: List[str]) -> str:
        """Create contextual prefix from header path"""
        if not header_path:
            return ""
        return "Document Context: " + " > ".join(header_path)


class JavaCodeChunker(BaseChunker):
    """
    AST-aware chunking for Java code
    Preserves class/method boundaries and Spring Boot annotations
    """

    def __init__(self):
        super().__init__()
        # Java-specific splitter
        self.code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA,
            chunk_size=rag_config.chunk_size_code,
            chunk_overlap=rag_config.chunk_overlap_code
        )

    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Chunk Java code using AST-aware splitting

        Args:
            content: Java source code
            metadata: File metadata

        Returns:
            List of Document objects with code-specific metadata
        """
        documents = []

        try:
            # Parse Java AST
            tree = javalang.parse.parse(content)
            package_name = tree.package.name if tree.package else "default"

            # Extract imports
            imports = [imp.path for imp in tree.imports] if tree.imports else []

            # Process each class in the file
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                class_name = node.name

                # Extract class-level annotations
                annotations = self._extract_annotations(node)

                # Get full class source
                class_start = node.position.line if node.position else 0
                class_source = self._extract_class_source(content, class_start)

                # Extract methods
                methods = []
                for method in node.methods:
                    methods.append(method.name)

                # Create metadata
                chunk_metadata = {
                    **metadata,
                    "doc_type": "code",
                    "language": "java",
                    "package": package_name,
                    "class_name": class_name,
                    "contains_methods": methods,
                    "annotations": annotations,
                    "imports": imports[:10],  # Limit to first 10
                    "chunk_id": f"{metadata.get('source', 'unknown')}_{class_name}",
                }

                # Create context prefix
                context_prefix = self._create_code_context_prefix(
                    package_name, class_name, annotations
                )

                # If class is too large, split by methods
                if len(class_source) > rag_config.chunk_size_code * 1.5:
                    method_chunks = self._split_by_methods(class_source, node)
                    for i, method_chunk in enumerate(method_chunks):
                        method_metadata = {
                            **chunk_metadata,
                            "chunk_id": f"{chunk_metadata['chunk_id']}_method_{i}",
                            "chunk_position": i,
                        }
                        enriched_content = f"{context_prefix}\n\n{method_chunk}"
                        documents.append(Document(
                            page_content=enriched_content,
                            metadata=method_metadata
                        ))
                else:
                    # Keep full class
                    enriched_content = f"{context_prefix}\n\n{class_source}"
                    documents.append(Document(
                        page_content=enriched_content,
                        metadata=chunk_metadata
                    ))

        except Exception as e:
            # Fallback to basic splitting if AST parsing fails
            print(f"AST parsing failed, using fallback: {e}")
            chunks = self.code_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    "doc_type": "code",
                    "language": "java",
                    "chunk_id": f"{metadata.get('source', 'unknown')}_chunk_{i}",
                    "chunk_position": i,
                    "parsing_method": "fallback",
                }
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))

        return documents

    def _extract_annotations(self, node) -> List[str]:
        """Extract Spring Boot and other annotations"""
        annotations = []
        if hasattr(node, 'annotations') and node.annotations:
            for ann in node.annotations:
                annotations.append(f"@{ann.name}")
        return annotations

    def _extract_class_source(self, content: str, start_line: int) -> str:
        """Extract full class source from content"""
        lines = content.split('\n')
        if start_line > 0:
            start_line -= 1  # 0-indexed

        # Simple heuristic: find class declaration and matching braces
        class_lines = []
        brace_count = 0
        started = False

        for i in range(start_line, len(lines)):
            line = lines[i]
            if 'class ' in line:
                started = True

            if started:
                class_lines.append(line)
                brace_count += line.count('{') - line.count('}')

                if brace_count == 0 and len(class_lines) > 1:
                    break

        return '\n'.join(class_lines)

    def _split_by_methods(self, class_source: str, class_node) -> List[str]:
        """Split class by methods"""
        # Simple split - could be enhanced with better AST traversal
        method_pattern = r'(public|private|protected)\s+[\w<>[\],\s]+\s+\w+\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{'

        chunks = []
        current_chunk = []
        brace_count = 0
        in_method = False

        for line in class_source.split('\n'):
            if re.search(method_pattern, line):
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                in_method = True

            current_chunk.append(line)
            brace_count += line.count('{') - line.count('}')

            if in_method and brace_count == 0 and len(current_chunk) > 1:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                in_method = False

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks if chunks else [class_source]

    def _create_code_context_prefix(
            self, package: str, class_name: str, annotations: List[str]
    ) -> str:
        """Create contextual prefix for code chunks"""
        prefix = f"Package: {package}\nClass: {class_name}"
        if annotations:
            prefix += f"\nAnnotations: {', '.join(annotations)}"
        return prefix


class SchemaChunker(BaseChunker):
    """
    Entity-relationship chunking for database schemas
    Preserves table definitions and relationships
    """

    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Chunk SQL schema by table definitions

        Args:
            content: SQL schema content
            metadata: File metadata

        Returns:
            List of Document objects with schema metadata
        """
        documents = []

        # Split by CREATE TABLE statements
        tables = re.split(
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?',
            content,
            flags=re.IGNORECASE
        )

        for i, table in enumerate(tables):
            if not table.strip():
                continue

            # Extract table name
            table_match = re.match(r'`?(\w+)`?', table)
            table_name = table_match.group(1) if table_match else f"table_{i}"

            # Extract columns
            columns = self._extract_columns(table)

            # Extract primary key
            primary_key = self._extract_primary_key(table)

            # Extract foreign keys
            foreign_keys = self._extract_foreign_keys(table)

            # Create metadata
            chunk_metadata = {
                **metadata,
                "doc_type": "schema",
                "table_name": table_name,
                "columns": columns,
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
                "chunk_id": f"{metadata.get('source', 'schema')}_{table_name}",
            }

            # Create context prefix
            context_prefix = f"Database Table: {table_name}"
            if columns:
                context_prefix += f"\nColumns: {', '.join(columns[:5])}"
                if len(columns) > 5:
                    context_prefix += f" (and {len(columns) - 5} more)"

            # Full table definition
            table_def = f"CREATE TABLE {table}"
            enriched_content = f"{context_prefix}\n\n{table_def}"

            documents.append(Document(
                page_content=enriched_content,
                metadata=chunk_metadata
            ))

        return documents

    def _extract_columns(self, table_def: str) -> List[str]:
        """Extract column names from table definition"""
        column_pattern = r'`?(\w+)`?\s+(?:VARCHAR|INT|BIGINT|TEXT|DATE|TIMESTAMP|DECIMAL|BOOLEAN|ENUM)'
        matches = re.findall(column_pattern, table_def, re.IGNORECASE)
        return matches

    def _extract_primary_key(self, table_def: str) -> Optional[str]:
        """Extract primary key"""
        pk_pattern = r'PRIMARY\s+KEY\s*\(`?(\w+)`?\)'
        match = re.search(pk_pattern, table_def, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_foreign_keys(self, table_def: str) -> List[Dict[str, str]]:
        """Extract foreign key relationships"""
        fk_pattern = r'FOREIGN\s+KEY\s*\(`?(\w+)`?\)\s*REFERENCES\s+`?(\w+)`?\s*\(`?(\w+)`?\)'
        matches = re.findall(fk_pattern, table_def, re.IGNORECASE)

        foreign_keys = []
        for match in matches:
            foreign_keys.append({
                "column": match[0],
                "ref_table": match[1],
                "ref_column": match[2]
            })
        return foreign_keys


# Factory function
def get_chunker(file_type: str) -> BaseChunker:
    """
    Get appropriate chunker based on file type

    Args:
        file_type: File extension (.md, .java, .sql, etc.)

    Returns:
        Appropriate chunker instance
    """
    chunker_map = {
        '.md': MarkdownChunker,
        '.markdown': MarkdownChunker,
        '.java': JavaCodeChunker,
        '.sql': SchemaChunker,
    }

    chunker_class = chunker_map.get(file_type.lower(), RecursiveCharacterTextSplitter)

    if chunker_class == RecursiveCharacterTextSplitter:
        # Default fallback chunker
        return RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
        )

    return chunker_class()


__all__ = [
    'BaseChunker',
    'MarkdownChunker',
    'JavaCodeChunker',
    'SchemaChunker',
    'get_chunker',
]