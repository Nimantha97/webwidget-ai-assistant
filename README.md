# webwidget-ai-assistant
IdeaBiz Web Widgets is a Spring MVC-based web application that provides embeddable registration and payment widgets for mobile subscription services.

WebWidget AI Chatbot - Production-Grade RAG System
A secure, local AI chatbot system for the WebWidget Java Spring Boot project that uses advanced RAG (Retrieval-Augmented Generation) to answer questions about the codebase, generate SQL queries, create reports, and assist with development tasks.

🎯 Key Features
Phase 1 (Core RAG)
Intelligent Q&A: Explain features, architecture, and development flows
Code Search: Semantic and keyword search across Java codebase
Documentation Retrieval: Find relevant info from markdown docs
Debugging Assistance: Analyze logs and troubleshoot issues
Phase 2 (Advanced Tools)
SQL Generation: Create read-only, parameterized queries
CSV Reports: Generate reports from live MySQL data
Code Generation: Suggest/create controllers, services, repositories
Bug Fixing: Analyze code and recommend fixes
Dynamic Ingestion: Upload new files for real-time knowledge updates
🏗️ Architecture
Hybrid RAG Pipeline
User Query
    ↓
Query Classification & Expansion
    ↓
┌─────────────┬─────────────┬─────────────┐
│   Vector    │   Keyword   │   Graph     │
│   (FAISS)   │   (BM25)    │   (Neo4j)   │
└──────┬──────┴──────┬──────┴──────┬──────┘
       └──────────┬──────────┘──────┘
                  ↓
       RRF Fusion (alpha-weighted)
                  ↓
       Cross-Encoder Reranking
                  ↓
       Top-5 Refined Chunks + Graph Paths
                  ↓
       Prompt Construction
                  ↓
       Qwen2.5-Coder-7B (4-bit quantized)
                  ↓
       LangChain Agent + Tools
                  ↓
       Generated Response
Multi-Modal Chunking
Documentation: Hierarchical header-based (MarkdownHeaderTextSplitter)
Java Code: AST-aware with class/method boundaries
DB Schemas: Entity-relationship based
Enrichment: Contextual prefixes, metadata, relationships
Technology Stack
LLM: Qwen2.5-Coder-7B-Instruct (4-bit quantized, ~4-5GB)
Backend: Python + FastAPI + LangChain
UI: Streamlit (demo), React/Redux/TypeScript (production)
Vector DB: ChromaDB (persistent, CPU-friendly)
Graph DB: Neo4j (relationships: Controller→Service→Repository)
Keyword Search: BM25 (via rank-bm25)
Embeddings: all-MiniLM-L6-v2 (SentenceTransformers)
Database: MySQL (local test DB)
History: SQLite
📁 Project Structure
webwidget-ai-chatbot/
├── app/
│   ├── api/                    # FastAPI routes
│   │   ├── __init__.py
│   │   ├── chat.py            # Chat endpoints
│   │   ├── upload.py          # File upload
│   │   └── session.py         # Session management
│   ├── rag/                    # RAG pipeline
│   │   ├── __init__.py
│   │   ├── ingestion.py       # Multi-modal ingestion
│   │   ├── retrieval.py       # Hybrid search + reranking
│   │   ├── chunkers.py        # Document/code/schema chunkers
│   │   └── query_processor.py # Query classification/expansion
│   ├── tools/                  # LangChain tools
│   │   ├── __init__.py
│   │   ├── sql_tool.py        # SQL generation/execution
│   │   ├── csv_tool.py        # Report generation
│   │   └── code_tool.py       # Code generation/suggestions
│   ├── graph/                  # Neo4j integration
│   │   ├── __init__.py
│   │   ├── builder.py         # Graph construction
│   │   └── retriever.py       # Graph-based retrieval
│   ├── memory/                 # Chat history
│   │   ├── __init__.py
│   │   └── history.py         # SQLite session storage
│   ├── models/                 # Pydantic models
│   │   ├── __init__.py
│   │   └── schemas.py         # Request/response DTOs
│   ├── config.py              # Configuration
│   └── __init__.py
├── data/
│   ├── codebase/              # Java source files
│   ├── docs/                  # Markdown/PDF documentation
│   ├── schemas/               # SQL schema dumps
│   ├── logs/                  # Application logs
│   ├── chroma/                # ChromaDB persistence
│   └── history.db             # Chat history SQLite
├── scripts/
│   ├── build_rag.py           # Initial data ingestion
│   ├── build_graph.py         # Neo4j graph construction
│   └── eval.py                # Evaluation metrics
├── tests/
│   ├── test_retrieval.py      # RAG tests
│   ├── test_tools.py          # Tool tests
│   └── test_queries.json      # Test dataset
├── ui/
│   ├── streamlit_app.py       # Streamlit demo UI
│   └── components/            # React components (future)
├── main.py                    # Application entry point
├── requirements.txt
├── config.yaml                # RAG configuration
├── .env.example
└── README.md
🚀 Setup Instructions
Prerequisites
Python 3.10+
MySQL 8.0+ (local instance)
Neo4j 5.0+ (Community Edition)
8GB+ RAM (for 4-bit quantized model)
Git
Installation
Clone and Setup Environment
git cl><repo-url>
cd webwidget-ai-chatbot
pyth>-m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Install System Dependencies
# Install Java parser
pip install javalang

# Install Neo4j (follow official docs for your OS)
# Ubuntu: sudo apt install neo4j
# macOS: brew install neo4j
Configure Environment
cp .env.example .env
# Edit .env with your settings:
# - MySQL credentials (localhost:3306, user: root, pass: 123)
# - Neo4j URI (bolt://localhost:7687)
# - Model path
Download LLM Model
# Using Hugging Face Transformers (automatic download on first run)
# Model: Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
# Will download ~4-5GB quantized model
Prepare Data
# Copy WebWidget project files
cp -r /path/to/webwidget/src/main/java/* data/codebase/
cp -r /path/to/webwidget/docs/* data/docs/
cp /path/to/schema.sql data/schemas/

# Or use provided sample structure
Initialize Databases
# Start Neo4j
neo4j start

# MySQL should be running with test data
mysql -u root -p123 ideabizadmin < data/schemas/schema.sql
Build RAG Pipeline
# Ingest all data (takes 5-10 minutes for moderate codebase)
python scripts/build_rag.py

# Build code relationship graph
python scripts/build_graph.py
Start Application
# Development mode with Streamlit UI
streamlit run ui/streamlit_app.py

# Production mode with FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
🔧 Configuration
Edit config.yaml to tune RAG parameters:

embeddings:
  model: "all-MiniLM-L6-v2"
  batch_size: 32

chunking:
  markdown:
    chunk_size: 800
    overlap: 150
  code:
    chunk_size: 1000
    overlap: 200

hybrid_search:
  vector_top_k: 20
  bm25_top_k: 20
  rrf_k: 60
  alpha_by_type:
    documentati>: 0.6  # Favor semantic
    code_search: 0.4    # Favor keywords
    schema: 0.3

reranking:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 5

llm:
  model_path: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
  quantizati>: "4bit"
  temperature: 0.1
  max_tokens: 2048
📊 Evaluation
Run evaluation suite to measure RAG accuracy:

pyth>--test-file tests/test_queries.json
Expected Metrics: - Recall@5: >0.80 (80% relevant docs in top 5) - MRR: >0.70 (first relevant in top 2-3) - NDCG@5: >0.75 - End-to-end accuracy: >85%

🎮 Usage Examples
Chat Interface (Streamlit)
``` User: Explain how the UserController handles authentication