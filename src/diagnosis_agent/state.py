from typing import TypedDict, Optional, Dict, Literal, List


class ImageInput(TypedDict):
    type: Literal["url", "base64", "path"]
    value: str


class EvaluationResult(TypedDict):
    metric: str
    score: float


class GraphState(TypedDict):
    question: str
    images: List[ImageInput] | None
    contexts: List[str] | None
    answer: str | None
    reasoning: List[str] | None
    evaluation: List[EvaluationResult] | None
