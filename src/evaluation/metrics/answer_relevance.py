from pydantic import BaseModel, Field
from ragas.prompt import PydanticPrompt
from dataclasses import dataclass, field
from ragas.metrics.base import MetricType, MetricWithLLM, SingleTurnMetric
import typing as t


class QARelevance(BaseModel):
    question: str = Field(description="The original question")
    answer: str = Field(description="The model's answer")


class AnswerRelevanceScore(BaseModel):
    score: float = Field(
        description="Relevance score between 0 and 1, where 1 means fully relevant"
    )
    reason: str = Field(description="Brief explanation for the score")


class AnswerRelevancePrompt(PydanticPrompt[QARelevance, AnswerRelevanceScore]):
    name: str = "answer_relevance"

    instruction: str = (
        "You are given a question and a model-generated answer.\n\n"
        "Evaluate how relevant the answer is to the question.\n\n"
        "Score relevance on a continuous scale from 0 to 1:\n"
        "- 1.0: Answer directly and fully addresses the question\n"
        "- 0.7–0.9: Mostly relevant, but missing details or slightly off-focus\n"
        "- 0.3–0.6: Partially relevant, addresses only a small part of the question\n"
        "- 0.0–0.2: Completely irrelevant or unrelated\n\n"
        "Judge semantic relevance, not factual correctness.\n"
        "Return JSON with a numeric score and a brief justification."
    )

    input_model = QARelevance
    output_model = AnswerRelevanceScore

    examples = [
        (
            QARelevance(
                question="What are the symptoms of pulmonary embolism?",
                answer="Pulmonary embolism can cause shortness of breath and chest pain.",
            ),
            AnswerRelevanceScore(
                score=0.9,
                reason="The answer directly addresses the symptoms, though not exhaustive.",
            ),
        ),
        (
            QARelevance(
                question="What causes malaria?",
                answer="Malaria is treated with antimalarial drugs.",
            ),
            AnswerRelevanceScore(
                score=0.2,
                reason="The answer discusses treatment, not the cause.",
            ),
        ),
    ]


@dataclass
class AnswerRelevance(MetricWithLLM, SingleTurnMetric):
    """
    Answer Relevance measures how relevant the model's answer is
    to the given question.

    Returns a continuous score in [0, 1].
    """

    name: str = "answer_relevance"

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"question", "response"},
        }
    )

    answer_relevance_prompt: PydanticPrompt = AnswerRelevancePrompt()

    async def _ascore(self, row):
        # Routing handled by SingleTurnMetric
        raise NotImplementedError

    async def _single_turn_ascore(self, sample, callbacks):
        prompt_input = QARelevance(
            question=sample.question,
            answer=sample.response,
        )

        prompt_response = await self.answer_relevance_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )

        score = float(prompt_response.score)

        # Optional debug logging
        print(
            {
                "answer_relevance_score": score,
                "reason": prompt_response.reason,
            }
        )

        return score
