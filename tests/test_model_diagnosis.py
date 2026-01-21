import pytest

from diagnosis_agent.state import GraphState
from diagnosis_agent.nodes.diagnosis_node import model_answer_and_reasoning


@pytest.mark.asyncio
async def test_model_answer_and_reasoning():
    state: GraphState = {
        "question": (
            "A 47-year-old man presents with sudden onset severe headache described "
            "as the worst headache of his life, associated with nausea and neck stiffness. "
            "CT head shows blood in the basal cisterns. Where is the bleeding located?"
        ),
        "contexts": [
            "Subarachnoid hemorrhage presents with thunderclap headache.",
            "Blood in basal cisterns indicates subarachnoid bleeding.",
        ],
        "images": None,
        "answer": None,
        "reasoning": None,
        "evaluation": None,
    }

    result = await model_answer_and_reasoning(state)

    # --- Assertions ---
    assert "answer" in result
    assert "reasoning" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["reasoning"], list)

    # --- Print outputs for inspection ---
    print("\n====== MODEL OUTPUT ======")
    print("Answer:")
    print(result["answer"])

    print("\nReasoning:")
    for i, r in enumerate(result["reasoning"], 1):
        print(f"{i}. {r}")
    print("==========================")
