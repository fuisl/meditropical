import pytest

from diagnosis_agent.state import GraphState
from diagnosis_agent.nodes.evaluation_node import evaluate_diagnosis


@pytest.mark.asyncio
async def test_evaluation_node_smoke():
    """
    Smoke test for evaluation node.
    Verifies structure, types, and prints metric scores.
    """

    state: GraphState = {
        "question": (
            "A 47-year-old man presents with sudden severe headache and neck stiffness. "
            "CT shows blood in the basal cisterns. Where is the bleeding located?"
        ),
        "contexts": [
            "Subarachnoid hemorrhage presents with thunderclap headache.",
            "Blood in basal cisterns indicates subarachnoid bleeding.",
        ],
        "answer": "Bleeding is located between the arachnoid mater and pia mater.",
        "reasoning": [
            "Thunderclap headache suggests subarachnoid hemorrhage.",
            "Basal cistern blood on CT supports this diagnosis.",
        ],
        "images": None,
        "evaluation": None,
    }

    result = await evaluate_diagnosis(state)

    # --- Assertions ---
    assert "evaluation" in result
    metrics = result["evaluation"]

    assert isinstance(metrics, list)
    assert len(metrics) > 0

    for m in metrics:
        assert isinstance(m, dict)
        assert "metric" in m
        assert "score" in m
        assert isinstance(m["metric"], str)
        assert isinstance(m["score"], float)

    # --- Print outputs for inspection ---
    print("\n====== EVALUATION METRICS ======")
    for m in metrics:
        print(f"{m['metric']}: {m['score']:.4f}")
    print("================================")
