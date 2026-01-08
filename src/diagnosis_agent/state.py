from typing import TypedDict, Optional, Dict


class GraphState(TypedDict):
    question: str
    contexts: list[str] | None
    answer: str | None
    reasoning: list[str] | None
    evaluation: Optional[Dict[str, float]]
