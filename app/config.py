"""
Configuration module for WebWidget AI Chatbot
Loads settings from config.yaml and .env
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # MySQL Configuration
    mysql_host: str = Field(default="localhost", env="MYSQL_HOST")
    mysql_port: int = Field(default=3306, env="MYSQL_PORT")
    mysql_database: str = Field(default="ideabizadmin", env="MYSQL_DATABASE")
    mysql_user: str = Field(default="root", env="MYSQL_USER")
    mysql_password: str = Field(default="123", env="MYSQL_PASSWORD")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")

    # Model Configuration
    model_cache_dir: str = Field(default=str(MODELS_DIR), env="MODEL_CACHE_DIR")
    hf_home: str = Field(default=str(MODELS_DIR / "huggingface"), env="HF_HOME")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")

    class Config:
        env_file = ".env"
        case_sensitive = False


class RAGConfig:
    """RAG pipeline configuration loaded from YAML"""

    def __init__(self, config_path: str = "Configuration/config.yaml"):
        config_file = ROOT_DIR / config_path
        with open(config_file, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    @property
    def embedding_model(self) -> str:
        return self.get('embeddings.model', 'all-MiniLM-L6-v2')

    @property
    def embedding_dimension(self) -> int:
        return self.get('embeddings.dimension', 384)

    @property
    def embedding_batch_size(self) -> int:
        return self.get('embeddings.batch_size', 32)

    @property
    def chunk_size_markdown(self) -> int:
        return self.get('chunking.markdown.chunk_size', 800)

    @property
    def chunk_overlap_markdown(self) -> int:
        return self.get('chunking.markdown.chunk_overlap', 150)

    @property
    def chunk_size_code(self) -> int:
        return self.get('chunking.code.chunk_size', 1000)

    @property
    def chunk_overlap_code(self) -> int:
        return self.get('chunking.code.chunk_overlap', 200)

    @property
    def vector_top_k(self) -> int:
        return self.get('hybrid_search.vector_top_k', 20)

    @property
    def bm25_top_k(self) -> int:
        return self.get('hybrid_search.bm25_top_k', 20)

    @property
    def graph_top_k(self) -> int:
        return self.get('hybrid_search.graph_top_k', 10)

    @property
    def rrf_k(self) -> int:
        return self.get('hybrid_search.rrf_k', 60)

    @property
    def default_alpha(self) -> float:
        return self.get('hybrid_search.default_alpha', 0.5)

    def get_alpha_for_query_type(self, query_type: str) -> float:
        """Get alpha value for specific query type"""
        alphas = self.get('hybrid_search.alpha_by_type', {})
        return alphas.get(query_type, self.default_alpha)

    @property
    def reranking_enabled(self) -> bool:
        return self.get('reranking.enabled', True)

    @property
    def reranking_model(self) -> str:
        return self.get('reranking.model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')

    @property
    def reranking_top_k(self) -> int:
        return self.get('reranking.top_k', 5)

    @property
    def llm_model_name(self) -> str:
        return self.get('llm.model_name', 'Qwen/Qwen2.5-Coder-7B-Instruct-GGUF')

    @property
    def llm_model_file(self) -> str:
        return self.get('llm.model_file', 'qwen2.5-coder-7b-instruct-q4_k_m.gguf')

    @property
    def llm_temperature(self) -> float:
        return self.get('llm.temperature', 0.1)

    @property
    def llm_max_tokens(self) -> int:
        return self.get('llm.max_tokens', 2048)

    @property
    def llm_context_window(self) -> int:
        return self.get('llm.context_window', 4096)

    @property
    def chroma_persist_dir(self) -> str:
        return str(DATA_DIR / self.get('vectorstore.persist_directory', 'data/chroma'))

    @property
    def chroma_collection_name(self) -> str:
        return self.get('vectorstore.collection_name', 'webwidget_knowledge')

    @property
    def history_db_path(self) -> str:
        return str(DATA_DIR / "history.db")

    @property
    def max_turns_per_session(self) -> int:
        return self.get('history.max_turns_per_session', 20)

    @property
    def graph_enabled(self) -> bool:
        return self.get('graph.enabled', True)

    @property
    def graph_relationship_types(self) -> List[str]:
        return self.get('graph.relationship_types', ['CALLS', 'USES', 'EXTENDS', 'QUERIES'])

    @property
    def caching_enabled(self) -> bool:
        return self.get('caching.enabled', True)

    @property
    def cache_max_size(self) -> int:
        return self.get('caching.max_size', 100)

    @property
    def cache_ttl(self) -> int:
        return self.get('caching.ttl', 3600)

    @property
    def query_expansion_enabled(self) -> bool:
        return self.get('query_processing.enable_expansion', True)

    @property
    def query_classification_enabled(self) -> bool:
        return self.get('query_processing.enable_classification', True)

    @property
    def query_patterns(self) -> Dict[str, List[str]]:
        return self.get('query_processing.patterns', {})


# Singleton instances
settings = Settings()
rag_config = RAGConfig()


# Database URLs
def get_mysql_url(async_driver: bool = False) -> str:
    """Get MySQL connection URL"""
    driver = "mysql+aiomysql" if async_driver else "mysql+pymysql"
    return (
        f"{driver}://{settings.mysql_user}:{settings.mysql_password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
    )


def get_neo4j_auth() -> tuple:
    """Get Neo4j authentication tuple"""
    return (settings.neo4j_user, settings.neo4j_password)


# Path helpers
def get_data_path(subdir: str) -> Path:
    """Get path to data subdirectory"""
    path = DATA_DIR / subdir
    path.mkdir(exist_ok=True, parents=True)
    return path


# Logging configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": settings.log_level,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "detailed",
            "level": "DEBUG",
        },
    },
    "root": {
        "level": settings.log_level,
        "handlers": ["console", "file"],
    },
}

# Export all
__all__ = [
    'settings',
    'rag_config',
    'ROOT_DIR',
    'DATA_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
    'get_mysql_url',
    'get_neo4j_auth',
    'get_data_path',
    'LOG_CONFIG',
]