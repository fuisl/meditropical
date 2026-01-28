from typing import Any, Dict

from diagnosis_agent.state import GraphState
from diagnosis_agent.utils import logger


async def get_user_input(state: GraphState) -> Dict[str, Any]:
    """
    Entry node.
    Validates and normalizes user input from LangSmith / LangGraph UI.
    """

    # --- Validate question ---
    question = state.get("question")
    if not question or not isinstance(question, str):
        raise ValueError("❌ `question` must be a non-empty string")

    question = question.strip()
    if not question:
        raise ValueError("❌ `question` cannot be empty")

    # --- Validate images ---
    images = state.get("images")
    validated_images = None

    if images is not None:
        if not isinstance(images, list):
            raise ValueError("❌ `images` must be a list of ImageInput objects")

        validated_images = []

        for i, img in enumerate(images):
            if not isinstance(img, dict):
                raise ValueError(f"❌ Image #{i} must be a dict")

            img_type = img.get("type")
            img_value = img.get("value")

            if img_type not in {"url", "base64", "path"}:
                raise ValueError(
                    f"❌ Image #{i} has invalid type '{img_type}'. "
                    "Must be one of: url, base64, path"
                )

            if not isinstance(img_value, str) or not img_value.strip():
                raise ValueError(f"❌ Image #{i} has empty or invalid value")

            validated_images.append(
                {
                    "type": img_type,
                    "value": img_value.strip(),
                }
            )

        logger.info("📷 Received %d validated image(s)", len(validated_images))

    logger.info("📝 User question received: %s", question[:200])

    # --- Return normalized state updates ---
    return {
        "question": question,
        "images": validated_images,
    }
