from typing import Any, Dict, Optional

from diagnosis_agent.state import GraphState


async def evaluate_diagnosis(state: GraphState) -> Dict[str, Any]:
    """
    Evaluate the diagnosis/answer using the metrics:
    - Answer Relevancy
    - Faithfulness
    - Context Precision
    - Context Recall


    Stores the computed floats in state['evaluation'].
    """
    question = state.get("question", "")
    answer = state.get("answer", "")
    contexts = state.get("contexts")

    # In a real implementation you might call separate LLM scorers here.
    answer_relevancy = 0.5  # Placeholder score
    faithfulness = 0.5  # Placeholder score
    context_precision = 0.5  # Placeholder score
    # For context recall you normally need gold contexts; here we reuse contexts as placeholders
    context_recall = 0.5  # Placeholder score

    evaluation = {
        "answer_relevancy": float(answer_relevancy),
        "faithfulness": float(faithfulness),
        "context_precision": float(context_precision),
        "context_recall": float(context_recall),
    }

    return {"evaluation": evaluation}
