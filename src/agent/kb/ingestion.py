"""
Document ingestion module for LightRAG + RAGAnything pipeline.

Handles end-to-end processing of multimodal documents (PDF, DOCX, PPTX, images)
and insertion into the knowledge base.
"""

import os
from typing import Optional, List, Dict, Any, Iterable
from pathlib import Path
from dataclasses import dataclass

from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status


@dataclass
class IngestionSettings:
    """Settings for document ingestion pipeline."""
    working_dir: str = os.getenv("WORKING_DIR", "./data/rag_storage")
    parser: str = os.getenv("PARSER", "mineru")  # mineru | docling
    parse_method: str = os.getenv("PARSE_METHOD", "auto")  # auto | ocr | txt
    enable_image: bool = True
    enable_table: bool = True
    enable_equation: bool = True
    output_dir: str = os.getenv("OUTPUT_DIR", "./data/parsed_output")


def make_raganything(
    lightrag_instance: LightRAG,
    settings: Optional[IngestionSettings] = None
) -> RAGAnything:
    """
    Factory to construct a RAGAnything instance wired to a LightRAG instance.
    
    Args:
        lightrag_instance: An initialized LightRAG instance
        settings: Optional ingestion settings (defaults to env-based config)
    
    Returns:
        RAGAnything instance configured for multimodal document processing
    """
    if settings is None:
        settings = IngestionSettings()
    
    # Extract the model functions from the LightRAG instance
    llm_func = lightrag_instance.llm_model_func
    embed_func = lightrag_instance.embedding_func
    
    # Validate that we have the required functions
    if llm_func is None:
        raise ValueError("LightRAG instance must have a configured llm_model_func")
    if embed_func is None:
        raise ValueError("LightRAG instance must have a configured embedding_func")
    
    # Create a vision model function from the LLM function
    # (assumes the LLM supports multimodal messages)
    def vision_model_func(prompt, system_prompt=None, history_messages=[], 
                         image_data=None, messages=None, **kwargs):
        if messages:
            return llm_func("", messages=messages, **kwargs)
        
        if image_data:
            msg_system = {"role": "system", "content": system_prompt} if system_prompt else None
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ]
            messages_local = [m for m in [msg_system, {"role": "user", "content": user_content}] if m]
            return llm_func("", messages=messages_local, **kwargs)
        
        return llm_func(prompt, system_prompt=system_prompt, 
                       history_messages=history_messages, **kwargs)
    
    return RAGAnything(
        config=RAGAnythingConfig(
            working_dir=settings.working_dir,
            parser=settings.parser,
            parse_method=settings.parse_method,
            enable_image_processing=settings.enable_image,
            enable_table_processing=settings.enable_table,
            enable_equation_processing=settings.enable_equation,
        ),
        lightrag=lightrag_instance,
        llm_model_func=llm_func,
        vision_model_func=vision_model_func,
        embedding_func=embed_func,
    )


async def process_document(
    file_path: str,
    rag_instance: RAGAnything,
    display_stats: bool = True,
    output_dir: Optional[str] = None,
    **parser_kwargs
) -> None:
    """
    End-to-end parse + insert for a single document.
    
    Produces structured blocks (text, images, tables, equations) and inserts
    them into LightRAG storages. DocStatus will track progress through
    PREPROCESSED → CHUNK → INDEX → READY states.
    
    Args:
        file_path: Path to document (PDF, DOCX, PPTX, images, etc.)
        rag_instance: Initialized RAGAnything instance
        display_stats: Whether to display processing statistics
        output_dir: Directory for parsed output artifacts (default: from settings)
        **parser_kwargs: Additional parser-specific options
            For MinerU: lang, device, start_page, end_page, formula, table, etc.
    """
    if output_dir is None:
        settings = IngestionSettings()
        output_dir = settings.output_dir
    
    await rag_instance.process_document_complete(
        file_path=file_path,
        output_dir=output_dir,
        display_stats=display_stats,
        **parser_kwargs
    )


async def insert_content_list(
    content_list: List[Dict[str, Any]],
    file_stub: str,
    rag_instance: RAGAnything,
    doc_id: Optional[str] = None,
    split_by_character: Optional[str] = None,
    split_by_character_only: bool = False,
    display_stats: bool = True,
) -> None:
    """
    Insert a manually-constructed multimodal content list.
    
    Useful when you have your own OCR/table extraction pipeline and want
    to insert the results directly into the knowledge base.
    
    Args:
        content_list: List of content dicts with structure:
            {"type": "text", "content": "...", "page_idx": 1}
            {"type": "image", "img_path": "/abs/path.png", "image_caption": [...], "page_idx": 1}
            {"type": "table", "content": "...", "page_idx": 2}
        file_stub: Provenance label (e.g., "report_2024.pdf")
        rag_instance: Initialized RAGAnything instance
        doc_id: Optional custom document ID
        split_by_character: Optional chunking size
        split_by_character_only: Whether to use only character-based splitting
        display_stats: Whether to display insertion statistics
        
    Note:
        All img_path values must be ABSOLUTE paths.
    """
    await rag_instance.insert_content_list(
        content_list=content_list,
        file_path=file_stub,
        doc_id=doc_id,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        display_stats=display_stats,
    )


async def insert_files(
    rag_instance: RAGAnything,
    files: List[Path],
    doc_ids: Optional[List[str]] = None,
    display_stats: bool = True,
    output_dir: Optional[str] = None,
    **parser_kwargs
) -> None:
    """
    Batch process multiple files.
    
    Args:
        rag_instance: Initialized RAGAnything instance
        files: List of file paths to process
        doc_ids: Optional list of custom document IDs (must match length of files)
        display_stats: Whether to display processing statistics
        output_dir: Directory for parsed output artifacts
        **parser_kwargs: Additional parser-specific options
    """
    if doc_ids and len(doc_ids) != len(files):
        raise ValueError("doc_ids must match the length of files if provided")
    
    for i, file_path in enumerate(files):
        doc_id = doc_ids[i] if doc_ids else None
        await process_document(
            file_path=str(file_path.absolute()),
            rag_instance=rag_instance,
            display_stats=display_stats,
            output_dir=output_dir,
            **parser_kwargs
        )
