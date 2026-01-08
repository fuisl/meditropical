from typing import Any, Dict, Optional

from diagnosis_agent.state import GraphState


async def model_diagnosis(state: GraphState) -> Dict[str, Any]:
    """
    Produce a diagnosis of the model's answer. The diagnosis should explain potential
    failure modes (hallucination, missing context, ambiguity, formatting issues, etc.).


    Expected output:
    - state['reasoning'] -> list of diagnosis points (strings)
    """
    # ----- TODO: call your explanatory LLM or chain here. Example placeholders: -----
    diag = [
        "The model used only top-2 contexts which may miss important supporting facts.",
        "Answer contains a confident assertion without citation — possible hallucination.",
    ]

    return {"reasoning": diag}
