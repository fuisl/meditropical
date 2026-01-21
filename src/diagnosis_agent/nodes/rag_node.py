import os
import dotenv
from typing import Any, Dict, Optional

import httpx
from diagnosis_agent.utils import logger

from diagnosis_agent.state import GraphState

dotenv.load_dotenv()


async def rag_retrieve(state: GraphState) -> Dict[str, Any]:

    question = state["question"]
    timeout = os.getenv("LIGHTRAG_TIMEOUT", 30.0)
    rag_result = await _generate_rag_response(
        question,
        client=httpx.AsyncClient(timeout=float(timeout)),
    )
    retrieved_contexts = rag_result["contexts"]
    return {"contexts": retrieved_contexts}


async def _generate_rag_response(
    question,
    client,
):
    try:
        rag_api_url = os.getenv("LIGHTRAG_API_URL", "http://localhost:9621")

        payload = {
            "query": question,
            "mode": "mix",
            "include_references": True,
            "include_chunk_content": True,  # NEW: Request chunk content in references
            "response_type": "Multiple Paragraphs",
            "top_k": int(os.getenv("EVAL_QUERY_TOP_K", "10")),
        }

        # Get API key from environment for authentication
        # api_key = os.getenv("LIGHTRAG_API_KEY")

        # Prepare headers with optional authentication
        headers = {}
        # if api_key:
        #     headers["X-API-Key"] = api_key

        # Single optimized API call - gets both answer AND chunk content
        response = await client.post(
            f"{rag_api_url}/query",
            json=payload,
            headers=headers if headers else None,
        )
        response.raise_for_status()
        result = response.json()

        answer = result.get("response", "No response generated")
        references = result.get("references", [])

        # DEBUG: Inspect the API response
        logger.debug("🔍 References Count: %s", len(references))
        if references:
            first_ref = references[0]
            logger.debug("🔍 First Reference Keys: %s", list(first_ref.keys()))
            if "content" in first_ref:
                content_preview = first_ref["content"]
                if isinstance(content_preview, list) and content_preview:
                    logger.debug(
                        "🔍 Content Preview (first chunk): %s...",
                        content_preview[0][:100],
                    )
                elif isinstance(content_preview, str):
                    logger.debug("🔍 Content Preview: %s...", content_preview[:100])

        # Extract chunk content from enriched references
        # Note: content is now a list of chunks per reference (one file may have multiple chunks)
        contexts = []
        for ref in references:
            content = ref.get("content", [])
            if isinstance(content, list):
                # Flatten the list: each chunk becomes a separate context
                contexts.extend(content)
            elif isinstance(content, str):
                # Backward compatibility: if content is still a string (shouldn't happen)
                contexts.append(content)

        return {
            "answer": answer,
            "contexts": contexts,  # List of strings from actual retrieved chunks
        }

    except httpx.ConnectError as e:
        raise Exception(
            f"❌ Cannot connect to LightRAG API at {rag_api_url}\n"
            f"   Make sure LightRAG server is running:\n"
            f"   python -m lightrag.api.lightrag_server\n"
            f"   Error: {str(e)}"
        )
    except httpx.HTTPStatusError as e:
        raise Exception(
            f"LightRAG API error {e.response.status_code}: {e.response.text}"
        )
    except httpx.ReadTimeout as e:
        raise Exception(
            f"Request timeout after waiting for response\n"
            f"   Question: {question[:100]}...\n"
            f"   Error: {str(e)}"
        )
    except Exception as e:
        raise Exception(f"Error calling LightRAG API: {type(e).__name__}: {str(e)}")
