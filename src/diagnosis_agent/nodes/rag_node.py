from typing import Any, Dict, Optional

from diagnosis_agent.state import GraphState


async def rag_retrieve_and_generate(state: GraphState) -> Dict[str, Any]:
    retrieved_contexts = [
        "<context 1: doc text or excerpt>",
        "<context 2: doc text or excerpt>",
    ]
    rag_answer = "<RAG generated answer goes here>"
    return {"contexts": retrieved_contexts, "answer": rag_answer}
