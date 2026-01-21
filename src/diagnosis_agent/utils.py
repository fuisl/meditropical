import re
import logging
import logging.handlers

# Initialize logger with basic configuration
logger = logging.getLogger("meditropical")
logger.propagate = False  # prevent log message send to root logger
logger.setLevel(logging.INFO)

# Add console handler if no handlers exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Set httpx logging level to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)

import base64
import httpx
from typing import List, Optional

from diagnosis_agent.state import ImageInput
from diagnosis_agent.prompt import FEW_SHOT_EXAMPLES


OLLAMA_URL = "http://localhost:11434/api/chat"


def prepare_images(images: Optional[List[ImageInput]]) -> Optional[List[str]]:
    """
    Convert ImageInput list into Ollama-compatible base64 images.
    """
    if not images:
        return None

    prepared: List[str] = []

    for img in images:
        if img["type"] == "base64":
            prepared.append(img["value"])

        elif img["type"] == "path":
            with open(img["value"], "rb") as f:
                prepared.append(base64.b64encode(f.read()).decode())

        elif img["type"] == "url":
            raise ValueError(
                "URL images must be downloaded and converted to base64 before use"
            )

        else:
            raise ValueError(f"Unsupported image type: {img['type']}")

    return prepared


async def ollama_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: Optional[List[str]] = None,
    timeout: float = 120.0,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
                **({"images": images} if images else {}),
            },
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def build_user_prompt(
    question: str,
    contexts: list[str] | None,
    images: List[ImageInput] | None,
) -> str:
    context_block = ""
    if contexts:
        context_block = "\n".join(f"- {c}" for c in contexts)

    image_block = ""
    if images:
        image_block = "\n".join(
            f"- Image ({img['type']}): {img['value']}" for img in images
        )

    return f"""
----------------------------------------
CASE PRESENTATION
----------------------------------------
Question:
{question}

Retrieved Context:
{context_block if context_block else "None"}

Images:
{image_block if image_block else "None"}

----------------------------------------
OUTPUT TEMPLATE
----------------------------------------
<think>
...your internal reasoning...
</think>
<answer>
...final answer only...
</answer>

----------------------------------------
EXAMPLES
----------------------------------------
{FEW_SHOT_EXAMPLES}
"""


def parse_think_answer(text: str) -> tuple[list[str], str]:
    think_match = re.search(r"<think>(.*?)</think>", text, re.S)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.S)

    if not think_match or not answer_match:
        raise ValueError("Model output does not match required format")

    reasoning_text = think_match.group(1).strip()
    answer_text = answer_match.group(1).strip()

    # Convert reasoning into bullet points if desired
    reasoning_lines = [
        line.strip("- ").strip() for line in reasoning_text.splitlines() if line.strip()
    ]

    return reasoning_lines, answer_text
