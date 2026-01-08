from typing import Any, Dict, Optional

from diagnosis_agent.state import GraphState


async def get_user_input(state: GraphState) -> Dict[str, Any]:
    """
    Entry node. The state is expected to already include the `question` field.
    You can use this node for pre-processing (cleaning question, spellcheck, logging).
    """
    # Example placeholder: ensure question exists
    if "question" not in state or not state.get("question"):
        raise ValueError("question must be set in the initial state")

    # return nothing (no-op) or return modifications
    return {}
