"""
FIXED Prompt Templates - Added max_length Parameter
"""
from typing import List
from app.models.schemas import RetrievedChunk, QueryType


def get_system_prompt(query_type: QueryType) -> str:
    """
    Get appropriate system prompt based on query type
    """
    base_prompt = """You are an expert AI assistant for the WebWidget Java Spring Boot application.

WebWidget Architecture:
- Spring Boot REST API with MySQL database
- Layered architecture: Controllers → Services → Repositories
- Key technologies: Spring Boot, JPA/Hibernate, MySQL, Maven

Your role:
- Answer questions about the codebase accurately
- Explain code functionality and architecture
- Help with debugging and troubleshooting
- Suggest improvements when asked

Important rules:
- ONLY use information from the provided context
- Always cite specific files/classes (e.g., "According to UserController.java...")
- If information isn't in the context, say so clearly
- Be concise and technical
- Use code snippets to illustrate points
"""

    type_prompts = {
        QueryType.CODE_SEARCH: """
Focus on:
- Explaining class/method functionality
- Showing code structure and annotations
- Identifying Spring Boot patterns (@RestController, @Service, etc.)
- Tracing method calls and dependencies
""",
        QueryType.DOCUMENTATION: """
Focus on:
- High-level architecture explanations
- Feature descriptions
- API documentation
- Design patterns and best practices
""",
        QueryType.SCHEMA: """
Focus on:
- Database schema details
- Table relationships and foreign keys
- Column definitions and constraints
- SQL queries and data access patterns
""",
        QueryType.DEBUGGING: """
Focus on:
- Identifying potential issues
- Explaining error causes
- Suggesting fixes based on codebase
- Relevant code locations to check
""",
        QueryType.GENERAL: """
Focus on:
- General codebase questions
- Feature explanations
- Architecture overview
"""
    }

    return base_prompt + type_prompts.get(query_type, type_prompts[QueryType.GENERAL])


def format_context(
    chunks: List[RetrievedChunk],
    max_chunks: int = 5,
    max_length: int = 4000  # ✅ FIXED: Added this parameter
) -> str:
    """
    Format retrieved chunks into context string

    Args:
        chunks: Retrieved context chunks
        max_chunks: Maximum number of chunks to include
        max_length: Maximum total character length (NEW)

    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant context found in the codebase."

    context_parts = []
    total_length = 0

    for i, chunk in enumerate(chunks[:max_chunks], 1):
        # Get source info
        source = chunk.source or "unknown"
        filename = chunk.metadata.get('filename', source.split('/')[-1])
        class_name = chunk.metadata.get('class_name', '')
        package = chunk.metadata.get('package', '')

        # Build header
        header = f"## Source {i}: {filename}"
        if class_name:
            header += f" (Class: {class_name})"
        if package:
            header += f"\nPackage: {package}"
        header += f"\nRelevance Score: {chunk.score:.3f}"

        # Build context entry
        context_entry = f"""{header}

```java
{chunk.content}
```
"""

        # ✅ FIX: Check length before adding
        entry_length = len(context_entry)
        if total_length + entry_length > max_length:
            # Truncate if needed
            remaining = max_length - total_length
            if remaining > 500:  # Only add if significant space left
                truncated_content = chunk.content[:remaining - 200]
                context_entry = f"""{header}

```java
{truncated_content}
... (truncated)
```
"""
                context_parts.append(context_entry)
            break

        context_parts.append(context_entry)
        total_length += entry_length

    result = "\n\n".join(context_parts)

    # Add summary if truncated
    if len(chunks) > len(context_parts):
        result += f"\n\n*(Showing {len(context_parts)} of {len(chunks)} relevant sources)*"

    return result


def format_code_snippet(code: str, language: str = "java") -> str:
    """Format code snippet for display"""
    return f"```{language}\n{code}\n```"


def create_citation(filename: str, class_name: str = None) -> str:
    """Create citation string for responses"""
    if class_name:
        return f"According to `{filename}` (class `{class_name}`)"
    return f"According to `{filename}`"


# Example prompts for different scenarios
EXPLAIN_FEATURE_PROMPT = """Explain how the {feature} feature works in WebWidget.
Include:
1. Main classes involved
2. Flow of execution
3. Key methods and their purposes
4. Database tables used (if applicable)
"""

DEBUG_ISSUE_PROMPT = """Help debug this issue: {issue}
Analyze:
1. Potential causes based on the code
2. Relevant code sections to check
3. Suggested fixes
4. Prevention strategies
"""

ARCHITECTURE_PROMPT = """Explain the architecture of {component}.
Cover:
1. Overall structure and design
2. Key classes and their responsibilities
3. Dependencies and relationships
4. Design patterns used
"""

CODE_REVIEW_PROMPT = """Review this code section: {code}
Provide:
1. Code quality assessment
2. Potential issues or bugs
3. Improvement suggestions
4. Best practices alignment
"""


__all__ = [
    'get_system_prompt',
    'format_context',
    'format_code_snippet',
    'create_citation',
    'EXPLAIN_FEATURE_PROMPT',
    'DEBUG_ISSUE_PROMPT',
    'ARCHITECTURE_PROMPT',
    'CODE_REVIEW_PROMPT',
]