from pydantic import BaseModel, Field
from typing import List
from ragas.prompt import PydanticPrompt

from dataclasses import dataclass, field
from ragas.metrics.base import MetricType
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric
import typing as t


class QA(BaseModel):
    ground_truth: str = Field(description="The correct answer")
    model_answer: str = Field(description="The model's predicted answer")


class AnswerAccuracyScore(BaseModel):
    score: float = Field(description="Answer correctness score between 0 and 1")
    reason: str = Field(description="Brief explanation for the assigned score")


class AnswerAccuracyPrompt(PydanticPrompt[QA, AnswerAccuracyScore]):
    name: str = "answer_accuracy"

    instruction: str = (
        "You are given a ground-truth correct answer, "
        "and a model's answer.\n\n"
        "Evaluate how correct the model's answer is compared to the ground truth.\n"
        "Return a correctness score between 0 and 1:\n"
        "- 1.0 means fully correct\n"
        "- 0.0 means completely incorrect\n"
        "- Partial scores allowed for partially correct answers\n\n"
        "The evaluation should be semantic, not lexical.\n"
        "Return JSON with a numeric score and a brief justification."
    )

    input_model = QA
    output_model = AnswerAccuracyScore

    examples = [
        (
            QA(
                ground_truth="acute appendicitis",
                model_answer="appendicitis",
            ),
            AnswerAccuracyScore(
                score=1.0, reason="The model answer correctly identifies the diagnosis."
            ),
        ),
        (
            QA(
                ground_truth="pulmonary embolism",
                model_answer="myocardial infarction",
            ),
            AnswerAccuracyScore(
                score=0.0, reason="The model answer identifies a different condition."
            ),
        ),
    ]


@dataclass
class AnswerAccuracy(MetricWithLLM, SingleTurnMetric):
    """
    Answer Accuracy measures whether the model's answer matches
    the ground-truth answer.

    The LLM produces a score in [0, 1].
    The final metric returns:
        1 if score >= threshold
        0 otherwise
    """

    name: str = "answer_accuracy"

    threshold: float = 0.8  # default threshold

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"reference", "response"},
        }
    )

    answer_accuracy_prompt: PydanticPrompt = AnswerAccuracyPrompt()

    async def _ascore(self, row):
        # routing handled by SingleTurnMetric
        raise NotImplementedError

    async def _single_turn_ascore(self, sample, callbacks):
        prompt_input = QA(
            ground_truth=sample.reference,
            model_answer=sample.response,
        )

        prompt_response = await self.answer_accuracy_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )

        raw_score = float(prompt_response.score)

        # Optional: log for inspection/debugging
        print(
            {
                "raw_score": raw_score,
                "threshold": self.threshold,
                "reason": prompt_response.reason,
            }
        )

        # Thresholding happens HERE
        return 1 if raw_score >= self.threshold else 0
