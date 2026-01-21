import os
import dotenv
import httpx
from typing import Any, Dict, List

from diagnosis_agent.state import GraphState
from diagnosis_agent.utils import (
    ollama_chat,
    prepare_images,
    build_user_prompt,
    parse_think_answer,
)
from diagnosis_agent.prompt import MEDCASE_SYSTEM_PROMPT
from diagnosis_agent.utils import logger

dotenv.load_dotenv()


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = os.getenv("LLM_MODEL", "gemma3:latest")


async def model_answer_and_reasoning(state: GraphState) -> Dict[str, Any]:
    prompt = build_user_prompt(
        question=state["question"],
        contexts=state.get("contexts"),
        images=state.get("images"),
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": MEDCASE_SYSTEM_PROMPT,
        "stream": False,
    }

    async with httpx.AsyncClient(
        timeout=float(os.getenv("LLM_TIMEOUT", 120))
    ) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

    raw_output = response.json()["response"]
    logger.debug("Raw model output:\n%s", raw_output)

    reasoning, answer = parse_think_answer(raw_output)

    return {
        "reasoning": reasoning,
        "answer": answer,
    }
