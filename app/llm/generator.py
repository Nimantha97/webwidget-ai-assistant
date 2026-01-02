"""
SPEED OPTIMIZED LLM Generator
Changes:
- Increased n_threads to match CPU cores
- Added n_batch optimization
- Reduced max_tokens for faster responses
- Optimized sampling parameters
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from llama_cpp import Llama
import multiprocessing

from app.config import rag_config
from app.models.schemas import RetrievedChunk, QueryType
from app.llm.prompts import get_system_prompt, format_context

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    SPEED OPTIMIZED LLM Generator
    Target: 10-15 seconds for typical responses on CPU
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize with SPEED-OPTIMIZED settings"""
        if model_path is None:
            model_path = Path("data/models/qwen-2.5-coder-7b/qwen2.5-coder-7b-instruct-q4_k_m.gguf")

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path.absolute()}\n"
                f"Please download the model and place it in the correct location."
            )

        # Detect CPU cores
        cpu_cores = multiprocessing.cpu_count()
        logger.info(f"Detected {cpu_cores} CPU cores")

        logger.info(f"Loading SPEED-OPTIMIZED LLM from: {self.model_path}")

        try:
            self.llm = Llama(
                model_path=str(self.model_path),

                # ✅ SPEED OPTIMIZATION 1: Use all CPU cores
                n_threads=cpu_cores,  # Use ALL cores (was 8)

                # ✅ SPEED OPTIMIZATION 2: Larger batch for throughput
                n_batch=512,  # Larger batches = faster processing

                # Context settings
                n_ctx=8192,   # ✅ REDUCED from 32768 (faster, still plenty)

                # Memory optimization
                use_mmap=True,
                use_mlock=False,

                # GPU (keep at 0 for CPU)
                n_gpu_layers=0,

                # RoPE settings for Qwen
                rope_freq_base=1000000,
                rope_scaling_type=1,

                verbose=False,
                seed=-1,
            )
            logger.info(f"✅ SPEED-OPTIMIZED LLM loaded")
            logger.info(f"   Threads: {cpu_cores}")
            logger.info(f"   Batch: 512")
            logger.info(f"   Context: 8192 tokens")

        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise

    def generate(
            self,
            query: str,
            retrieved_chunks: List[RetrievedChunk],
            query_type: QueryType = QueryType.GENERAL,
            chat_history: Optional[List[Dict[str, str]]] = None,
            temperature: float = 0.1,
            max_tokens: int = 512  # ✅ REDUCED from 1024 for speed
    ) -> str:
        """
        Generate response with SPEED OPTIMIZATIONS
        Target: 10-15 seconds on modern CPU
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🤖 GENERATING RESPONSE")
        logger.info(f"{'='*80}")

        # Filter chunks (use 0.001 threshold to match retrieval)
        filtered_chunks = self._filter_and_limit_chunks(
            retrieved_chunks,
            max_chunks=3,
            min_score=0.001
        )

        if not filtered_chunks:
            logger.warning("❌ No chunks passed threshold!")
            return (
                "I couldn't find highly relevant information about that in the codebase. "
                "Try rephrasing your question or being more specific."
            )

        # Build COMPACT context (speed optimization)
        context = format_context(
            filtered_chunks,
            max_chunks=3,
            max_length=2000  # ✅ REDUCED from 4000 for speed
        )

        # Get system prompt
        system_prompt = get_system_prompt(query_type)

        # Build messages
        messages = self._build_messages_compact(
            system_prompt=system_prompt,
            context=context,
            query=query,
            chat_history=chat_history
        )

        # Verify token count
        total_tokens = self._estimate_tokens(messages)
        logger.info(f"📊 Estimated context tokens: {total_tokens}")

        if total_tokens > 6000:  # Leave room for response
            logger.warning(f"⚠️ Context large ({total_tokens} tokens), truncating...")
            filtered_chunks = filtered_chunks[:2]
            context = format_context(filtered_chunks, max_chunks=2, max_length=1500)
            messages = self._build_messages_compact(system_prompt, context, query, None)

        # Generate with SPEED-OPTIMIZED parameters
        try:
            logger.info("⏳ Generating (optimized for speed)...")
            import time
            start_time = time.time()

            response = self.llm.create_chat_completion(
                messages=messages,

                # ✅ SPEED OPTIMIZATION 3: Faster sampling
                temperature=temperature,
                max_tokens=max_tokens,

                # ✅ SPEED OPTIMIZATION 4: Reduced sampling complexity
                top_p=0.9,        # Slightly lower (was 0.95)
                top_k=40,         # Keep this
                repeat_penalty=1.1,

                # Stop tokens
                stop=["</s>", "<|endoftext|>", "User:", "Question:", "\n\nUser:", "\n\nQuestion:"],
            )

            generated_text = response['choices'][0]['message']['content']

            # Log performance
            elapsed = time.time() - start_time
            tokens_generated = response['usage'].get('completion_tokens', 0) if 'usage' in response else len(generated_text.split())
            tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

            logger.info(f"✅ Generated {tokens_generated} tokens in {elapsed:.1f}s")
            logger.info(f"   Speed: {tokens_per_sec:.1f} tokens/sec")

            if elapsed > 30:
                logger.warning(f"⚠️ Generation took {elapsed:.1f}s (should be ~10-15s)")
                logger.warning(f"   Consider: Reduce max_tokens or context length")

            logger.info(f"{'='*80}\n")
            return generated_text.strip()

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")

            # Fallback: try with minimal context
            if "exceed" in str(e).lower() or "context" in str(e).lower():
                logger.info("🔄 Retrying with minimal context...")
                return self._generate_with_minimal_context(query, filtered_chunks[0])

            return f"Error generating response: {str(e)}"

    def _filter_and_limit_chunks(
        self,
        chunks: List[RetrievedChunk],
        max_chunks: int = 3,
        min_score: float = 0.001
    ) -> List[RetrievedChunk]:
        """Filter chunks with realistic threshold for RRF scores"""
        logger.info(f"\n📋 Filtering {len(chunks)} chunks (threshold: {min_score})")

        # Filter by minimum score
        filtered = [c for c in chunks if c.score >= min_score]

        if not filtered:
            logger.warning(f"   ⚠️ No chunks >= {min_score}, taking top 2 anyway")
            filtered = chunks[:2]

        # Sort by score descending
        filtered.sort(key=lambda x: x.score, reverse=True)

        # Limit to max_chunks
        filtered = filtered[:max_chunks]

        logger.info(f"   ✅ Selected {len(filtered)} chunks")
        for i, c in enumerate(filtered, 1):
            filename = c.metadata.get('filename', 'unknown')
            logger.info(f"      {i}. {filename} (score: {c.score:.4f})")

        return filtered

    def _build_messages_compact(
            self,
            system_prompt: str,
            context: str,
            query: str,
            chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Build COMPACT messages for faster generation
        """
        messages = []

        # ✅ ULTRA-COMPACT system prompt for speed
        compact_system = """You are a WebWidget (Java Spring Boot) code assistant.
Rules: Answer from context only. Cite files. Be concise."""

        messages.append({"role": "system", "content": compact_system})

        # NO chat history for speed (add later if needed)

        # Compact user message
        user_message = f"""Context:
{context}

Q: {query}
A:"""

        messages.append({"role": "user", "content": user_message})

        return messages

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count (rough: 1 token ≈ 4 chars)"""
        total_chars = sum(len(m['content']) for m in messages)
        return total_chars // 4

    def _generate_with_minimal_context(
        self,
        query: str,
        chunk: RetrievedChunk
    ) -> str:
        """Fallback: generate with just one chunk"""
        logger.info("🔄 Fallback mode: using single best chunk")

        messages = [
            {"role": "system", "content": "Answer based on code."},
            {"role": "user", "content": f"Code:\n{chunk.content[:800]}\n\nQ: {query}\nA:"}
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=256,
            stop=["</s>"]
        )

        return response['choices'][0]['message']['content'].strip()


# Singleton
_generator_instance = None


def get_generator(model_path: Optional[str] = None) -> LLMGenerator:
    """Get or create speed-optimized generator"""
    global _generator_instance

    if _generator_instance is None:
        _generator_instance = LLMGenerator(model_path)

    return _generator_instance


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        generator = get_generator()
        logger.info("✅ Speed-optimized generator ready")

        # Test
        from app.models.schemas import RetrievedChunk, QueryType
        test_chunks = [
            RetrievedChunk(
                content="@RestController\npublic class UserController {\n  @GetMapping('/users')\n  public List<User> getUsers() {...}\n}",
                score=0.95,
                source="UserController.java",
                metadata={"class_name": "UserController", "filename": "UserController.java"},
                chunk_id="test_1"
            )
        ]

        import time
        start = time.time()

        response = generator.generate(
            query="What does UserController do?",
            retrieved_chunks=test_chunks,
            query_type=QueryType.CODE_SEARCH
        )

        elapsed = time.time() - start

        print(f"\n✅ Test Response ({elapsed:.1f}s):")
        print(response)

        if elapsed < 20:
            print(f"\n✅ Speed is GOOD ({elapsed:.1f}s)")
        else:
            print(f"\n⚠️ Speed is SLOW ({elapsed:.1f}s - should be ~10-15s)")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)