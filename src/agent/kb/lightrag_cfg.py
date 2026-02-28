# kb/lightrag_cfg.py
from pydantic import BaseModel
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.rerank import cohere_rerank
import os, asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import Callable, Any, Optional, Tuple
from raganything import RAGAnything, RAGAnythingConfig

# Get the repository root directory (3 levels up from this file)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data"

# Load environment variables from project .env (do not override existing env vars)
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

class LightRAGConfig(BaseModel):
    """Configuration for building a LightRAG instance. Values default from environment when available."""
    input_dir: str = os.getenv("INPUT_DIR", str(DATA_DIR / "inputs"))
    working_dir: str = os.getenv("WORKING_DIR", str(DATA_DIR / "rag_storage"))
    kv_storage: str = os.getenv("LIGHTRAG_KV_STORAGE", "PGKVStorage")
    vector_storage: str = os.getenv("LIGHTRAG_VECTOR_STORAGE", "QdrantVectorDBStorage")
    graph_storage: str = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
    doc_status_storage: str = os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "PGDocStatusStorage")

    # External stores (fall back to env values)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", "")

    # Models: swap to your stack (env overrides)
    embedding_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    rerank_provider: str = os.getenv("RERANK_BINDING", "cohere")

async def build_lightrag(cfg: LightRAGConfig) -> LightRAG:
    """Async builder that initializes storages and returns a configured LightRAG instance."""
    setup_logger("lightrag")

    from lightrag.llm.openai import openai_complete_if_cache, openai_embed

    # llm wrapper: matches the signature used by test2.py (can be async or sync caller)
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kw):
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return await openai_complete_if_cache(
            model=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST"),
            **kw,
        )

    async def embedding_func(texts: list[str]):
        embed_model = os.getenv("EMBEDDING_MODEL", cfg.embedding_name)
        return await openai_embed(
            texts,
            model=embed_model,
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY", os.getenv("OPENAI_API_KEY")),
            base_url=os.getenv("EMBEDDING_BINDING_HOST"),
        )

    rag = LightRAG(
        working_dir=cfg.working_dir,
        kv_storage=cfg.kv_storage,
        vector_storage=cfg.vector_storage,
        graph_storage=cfg.graph_storage,
        doc_status_storage=cfg.doc_status_storage,
        vector_db_storage_cls_kwargs={
            "url": cfg.QDRANT_URL, "api_key": cfg.QDRANT_API_KEY
        } if cfg.vector_storage == "QdrantVectorDBStorage" else {},
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(embedding_dim=cfg.embedding_dim, func=embedding_func),
    )

    # attach a simple reranker if available
    rag.rerank_model_func = cohere_rerank

    # await rag.initialize_storages()  # will be called by RAGAnything
    # await initialize_pipeline_status()
    return rag


def _make_vision_model_func(rag: LightRAG) -> Callable[..., Any]:
    """Create a vision_model_func compatible with RAGAnything's expectation in test2.py.

    The function supports the `messages` form (multimodal) and `image_data` base64 inline format.
    If no vision-specific LLM is desired, it falls back to the configured `rag.llm_model_func`.
    """

    llm_callable = rag.llm_model_func

    if not callable(llm_callable):
        def _missing_llm(*args, **kwargs):
            raise RuntimeError("LightRAG instance does not have a callable llm_model_func configured")

        llm_callable = _missing_llm

    def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        # If messages provided (already in chat format), use LLM directly
        if messages:
            return llm_callable(
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                **kwargs,
            )

        if image_data:
            # Construct messages list expected by openai-style multimodal helper
            msg_system = {"role": "system", "content": system_prompt} if system_prompt else None
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ]
            messages_local = [m for m in [msg_system, {"role": "user", "content": user_content}] if m]
            return llm_callable("", system_prompt=None, history_messages=[], messages=messages_local, **kwargs)

        # fallback to plain text
        return llm_callable(prompt, system_prompt=system_prompt, history_messages=history_messages, **kwargs)

    return vision_model_func


async def load_lightrag_from_env() -> tuple[LightRAG, Callable[..., Any]]:
    """Convenience async loader: builds a LightRAG from env/defaults and returns (lightrag, vision_model_func).

    Example usage in test2.py:
        lightrag_instance, vision_model = await load_lightrag_from_env()
        rag = RAGAnything(lightrag=lightrag_instance, vision_model_func=vision_model)
    """
    cfg = LightRAGConfig()
    rag = await build_lightrag(cfg)
    vision = _make_vision_model_func(rag)
    return rag, vision

async def build_raganything(cfg: Optional[LightRAGConfig] = None) -> RAGAnything:
    """
    Build a fully initialized RAGAnything instance from configuration.
    
    This is the recommended entrypoint for the agent KB pipeline.
    Returns a RAGAnything instance ready for document processing.
    
    Args:
        cfg: Optional LightRAGConfig (defaults to env-based config)
    
    Returns:
        Initialized RAGAnything instance
    """
    if cfg is None:
        cfg = LightRAGConfig()
    
    # Build and initialize LightRAG
    lightrag = await build_lightrag(cfg)
    await lightrag.initialize_storages()
    await initialize_pipeline_status()
    
    # Create vision model function
    vision = _make_vision_model_func(lightrag)
    
    # Import here to avoid circular dependency
    from .ingestion import make_raganything as _make_rag
    
    # Build RAGAnything using the ingestion factory
    rag_anything = _make_rag(lightrag)
    
    return rag_anything


async def get_initialized_rag(cfg: Optional[LightRAGConfig] = None) -> tuple[RAGAnything, LightRAG]:
    """
    Create and initialize LightRAG and RAGAnything, returning (rag, lightrag).

    This is the recommended entrypoint for external callers who want a ready-to-use
    RAGAnything instance following the ingestion pipeline.
    
    Args:
        cfg: Optional LightRAGConfig (defaults to env-based config)
    
    Returns:
        Tuple of (RAGAnything instance, LightRAG instance)
    """
    if cfg is None:
        cfg = LightRAGConfig()
    
    lightrag = await build_lightrag(cfg)
    await lightrag.initialize_storages()
    await initialize_pipeline_status()
    
    from .ingestion import make_raganything as _make_rag
    rag_instance = _make_rag(lightrag)
    
    return rag_instance, lightrag