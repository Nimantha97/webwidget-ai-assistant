"""
Streamlit UI - UPDATED with Smart Retrieval & Metadata Tools
Now handles "list all" queries correctly!
"""
import streamlit as st
import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import smart retrieval and metadata tools
try:
    from app.rag.smart_retrieval import get_smart_retriever
    from app.rag.metadata_tools import get_lister
    SMART_RETRIEVAL = True
except ImportError:
    # Fallback to regular retrieval
    from app.rag.retrieval import HybridRetriever
    SMART_RETRIEVAL = False

from app.rag.query_processor import QueryProcessor
from app.llm.generator import get_generator
from app.models.schemas import QueryType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="WebWidget AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 WebWidget AI Assistant")
st.markdown("**OPTIMIZED** - Smart retrieval + Fast generation + Metadata tools")


# Initialize components
@st.cache_resource
def load_components():
    """Load retriever, processor, metadata lister, and LLM"""
    try:
        # Load retriever (smart or regular)
        if SMART_RETRIEVAL:
            retriever = get_smart_retriever()
            st.info("✅ Smart Retrieval enabled (adaptive top_k)")
        else:
            retriever = HybridRetriever()
            st.info("✅ Regular retrieval")

        processor = QueryProcessor()

        # Load metadata lister
        try:
            lister = get_lister()
            st.success("✅ Metadata tools enabled (fast 'list all' queries)")
        except Exception as e:
            logger.warning(f"Metadata lister failed: {e}")
            lister = None

        # Load LLM
        with st.spinner("Loading AI model..."):
            generator = get_generator()

        return retriever, processor, lister, generator

    except Exception as e:
        st.error(f"Failed to load components: {e}")
        st.stop()


# Load components
try:
    retriever, processor, lister, generator = load_components()
    st.success("✅ AI Assistant ready! (Optimized for speed)")
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources Used"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{i}. {source['file']}** (relevance: {source['score']:.1%})")
                    with st.container():
                        st.code(source['content'][:300] + "...", language="java")

# Chat input
if prompt := st.chat_input("Ask about your WebWidget project..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.chat_message("assistant"):
        # =====================================================================
        # STEP 1: Check if this is a "list all" metadata query
        # =====================================================================
        list_patterns = [
            (r'\ball controllers?\b', 'controllers'),
            (r'\blist (?:all )?controllers?\b', 'controllers'),
            (r'\bshow (?:me )?(?:all )?controllers?\b', 'controllers'),
            (r'\ball services?\b', 'services'),
            (r'\ball repositories?\b', 'repositories'),
            (r'\bproject structure\b', 'structure'),
        ]

        is_metadata_query = False
        metadata_type = None

        if lister:
            prompt_lower = prompt.lower()
            for pattern, qtype in list_patterns:
                if re.search(pattern, prompt_lower):
                    is_metadata_query = True
                    metadata_type = qtype
                    break

        # =====================================================================
        # Handle metadata queries (FAST - no LLM needed!)
        # =====================================================================
        if is_metadata_query:
            with st.spinner("🔍 Querying project metadata..."):
                try:
                    if metadata_type == 'controllers':
                        controllers = lister.list_all_controllers()

                        response = f"**Found {len(controllers)} controllers in the project:**\n\n"

                        for i, ctrl in enumerate(controllers, 1):
                            response += f"{i}. **{ctrl['filename']}**\n"
                            if ctrl['class_name']:
                                response += f"   - Class: `{ctrl['class_name']}`\n"
                            if ctrl['package']:
                                response += f"   - Package: `{ctrl['package']}`\n"
                            response += "\n"

                        st.markdown(response)

                        # Store
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": [{'file': c['filename'], 'score': 1.0, 'content': ''} for c in controllers]
                        })

                    elif metadata_type == 'services':
                        services = lister.list_all_services()
                        response = f"**Found {len(services)} services:**\n\n"
                        for i, svc in enumerate(services, 1):
                            response += f"{i}. `{svc['filename']}`\n"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    elif metadata_type == 'repositories':
                        repos = lister.list_all_repositories()
                        response = f"**Found {len(repos)} repositories/DAOs:**\n\n"
                        for i, repo in enumerate(repos, 1):
                            response += f"{i}. `{repo['filename']}`\n"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    elif metadata_type == 'structure':
                        structure = lister.get_project_structure()

                        response = "**WebWidget Project Structure:**\n\n"

                        response += "```\n"
                        response += "src/main/java/com/dialog/ideabizweb/\n"

                        if structure['controllers']:
                            response += "├── controllers/\n"
                            for ctrl in structure['controllers'][:15]:
                                response += f"│   ├── {ctrl}\n"
                            if len(structure['controllers']) > 15:
                                response += f"│   └── ... and {len(structure['controllers']) - 15} more\n"

                        if structure['services']:
                            response += "├── services/\n"
                            for svc in structure['services'][:10]:
                                response += f"│   ├── {svc}\n"
                            if len(structure['services']) > 10:
                                response += f"│   └── ... and {len(structure['services']) - 10} more\n"

                        if structure['repositories']:
                            response += "├── repositories/\n"
                            for repo in structure['repositories'][:10]:
                                response += f"│   ├── {repo}\n"
                            if len(structure['repositories']) > 10:
                                response += f"│   └── ... and {len(structure['repositories']) - 10} more\n"

                        response += "```\n"

                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"Metadata query failed: {e}")

        # =====================================================================
        # Handle regular RAG queries
        # =====================================================================
        else:
            with st.spinner("🔍 Searching codebase..."):
                try:
                    # Process query
                    processed_query, query_type, expansions = processor.process(prompt)

                    # Sidebar info
                    with st.sidebar:
                        st.info(f"**Query Type:** {query_type.value}")
                        if expansions:
                            st.caption(f"Expanded terms: {', '.join(expansions)}")

                    # Retrieve (smart retrieval adapts top_k automatically)
                    chunks = retriever.retrieve(
                        query=processed_query,
                        query_type=query_type,
                        use_graph=False
                    )

                    if chunks:
                        # Generate response with LLM
                        with st.spinner("💭 Generating response..."):
                            response = generator.generate(
                                query=prompt,
                                retrieved_chunks=chunks,
                                query_type=query_type,
                                chat_history=[
                                    {"role": msg["role"], "content": msg["content"]}
                                    for msg in st.session_state.messages[-4:-1]
                                ],
                                temperature=0.1,
                                max_tokens=512  # Optimized for speed
                            )

                        # Display response
                        st.markdown(response)

                        # Prepare sources
                        sources = [
                            {
                                "file": chunk.metadata.get('filename', 'unknown'),
                                "score": chunk.score,
                                "content": chunk.content
                            }
                            for chunk in chunks[:5]
                        ]

                        # Show sources
                        with st.expander("📚 Sources Used", expanded=False):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}. {source['file']}** (relevance: {source['score']:.1%})")
                                st.code(source['content'][:300] + "...", language="java")

                        # Add to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })
                    else:
                        # No relevant chunks found
                        response = (
                            "❌ I couldn't find relevant information about that. "
                            "Try rephrasing your question or asking about a different topic."
                        )
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": []
                        })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Error processing query: {e}", exc_info=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")

    st.markdown("""
    ### 🎯 How to Use
    
    **Fast Metadata Queries** (instant!):
    - "List all controllers"
    - "Show me all services"
    - "What's the project structure?"
    
    **AI-Powered Queries** (10-15s):
    - "What does UserController do?"
    - "Explain authentication flow"
    - "How are orders processed?"
    
    ### 💡 Tips
    - Use "list all" for complete listings
    - Ask specific questions for detailed explanations
    - Be clear and concise
    """)

    st.divider()

    # Statistics
    st.header("📊 Statistics")
    st.metric("Messages", len(st.session_state.messages))

    try:
        if lister:
            structure = lister.get_project_structure()
            st.metric("Controllers", len(structure['controllers']))
            st.metric("Services", len(structure.get('services', [])))
            st.metric("Repositories", len(structure.get('repositories', [])))
    except:
        pass

    st.divider()

    # Actions
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("WebWidget AI Assistant • Optimized for Speed")