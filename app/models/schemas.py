"""
Pydantic models for API request/response schemas
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class QueryType(str, Enum):
    """Query classification types"""
    DOCUMENTATION = "documentation"
    CODE_SEARCH = "code_search"
    SCHEMA = "schema"
    DEBUGGING = "debugging"
    GENERAL = "general"


class MessageRole(str, Enum):
    """Chat message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Single chat message"""
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Chat API request"""
    message: str = Field(..., description="User query message")
    session_id: Optional[str] = Field(None, description="Session ID for history")
    include_history: bool = Field(default=True, description="Include chat history")
    query_type: Optional[QueryType] = Field(None, description="Override auto-detected query type")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Explain how UserController handles authentication",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "include_history": True
            }
        }


class RetrievedChunk(BaseModel):
    """Retrieved document chunk"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    chunk_id: str


class GraphPath(BaseModel):
    """Graph relationship path"""
    path: List[str]
    relationship_types: List[str]
    description: str


class ChatResponse(BaseModel):
    """Chat API response"""
    response: str
    session_id: str
    query_type: QueryType
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    graph_paths: Optional[List[GraphPath]] = None
    tool_calls: Optional[List[str]] = None
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "response": "The UserController handles authentication through...",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "query_type": "code_search",
                "processing_time_ms": 1234.56
            }
        }


class SessionCreate(BaseModel):
    """Create new chat session"""
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Session creation response"""
    session_id: str
    created_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class SessionHistory(BaseModel):
    """Session history"""
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime
    last_updated: datetime


class FileUploadRequest(BaseModel):
    """File upload metadata"""
    filename: str
    file_type: str
    session_id: Optional[str] = None


class FileUploadResponse(BaseModel):
    """File upload response"""
    success: bool
    filename: str
    chunks_added: int
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "filename": "new_feature_docs.md",
                "chunks_added": 15,
                "message": "File successfully ingested into knowledge base"
            }
        }


class SQLGenerationRequest(BaseModel):
    """SQL generation request"""
    query_description: str
    table_hints: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "query_description": "Get monthly report for users registered in 2024",
                "table_hints": ["users", "registrations"]
            }
        }


class SQLGenerationResponse(BaseModel):
    """SQL generation response"""
    sql_query: str
    explanation: str
    tables_used: List[str]
    is_safe: bool
    warnings: Optional[List[str]] = None


class CSVReportRequest(BaseModel):
    """CSV report generation request"""
    query_description: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "query_description": "Generate monthly user registration report",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }


class CSVReportResponse(BaseModel):
    """CSV report generation response"""
    success: bool
    filename: str
    download_url: str
    row_count: int
    columns: List[str]
    preview: Optional[List[Dict[str, Any]]] = None


class CodeGenerationRequest(BaseModel):
    """Code generation request"""
    description: str
    code_type: str = Field(..., description="controller, service, repository, dto")
    class_name: str
    package: Optional[str] = None
    dependencies: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "description": "Create a service to handle product inventory",
                "code_type": "service",
                "class_name": "InventoryService",
                "package": "com.webwidget.services",
                "dependencies": ["InventoryRepository"]
            }
        }


class CodeGenerationResponse(BaseModel):
    """Code generation response"""
    success: bool
    code: str
    filename: str
    explanation: str
    dependencies: List[str]
    warnings: Optional[List[str]] = None


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, str]

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 12345.67,
                "components": {
                    "database": "connected",
                    "vector_store": "ready",
                    "llm": "loaded"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid request",
                "detail": "Session ID not found",
                "error_code": "SESSION_NOT_FOUND"
            }
        }


# Export all
__all__ = [
    'QueryType',
    'MessageRole',
    'ChatMessage',
    'ChatRequest',
    'ChatResponse',
    'RetrievedChunk',
    'GraphPath',
    'SessionCreate',
    'SessionResponse',
    'SessionHistory',
    'FileUploadRequest',
    'FileUploadResponse',
    'SQLGenerationRequest',
    'SQLGenerationResponse',
    'CSVReportRequest',
    'CSVReportResponse',
    'CodeGenerationRequest',
    'CodeGenerationResponse',
    'HealthCheck',
    'ErrorResponse',
]