from pydantic import BaseModel, Field
from typing import List
from ragas.prompt import PydanticPrompt

from dataclasses import dataclass, field
from ragas.metrics.base import MetricType
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric
import typing as t


class QCR(BaseModel):
    ground_truth_reasoning: str = Field(description="Ground-truth reasoning")
    model_reasoning: str = Field(description="The reasoning text produced by the model")


class ReasoningRecallClassification(BaseModel):
    statement: str = Field(
        description="A single sentence or step extracted from the reasoning"
    )
    reason: str = Field(
        description="Explanation of why the statement is or is not supported by the context"
    )
    attributed: int = Field(
        description="Binary label: 1 if the reasoning sentence is supported by the context, otherwise 0"
    )


class ReasoningRecallClassifications(BaseModel):
    classifications: List[ReasoningRecallClassification]


class ReasoningRecallClassificationPrompt(
    PydanticPrompt[QCR, ReasoningRecallClassifications]
):
    name: str = "reasoning_recall_classification"

    instruction: str = (
        "You are given a list of ground-truth reasons from a case report and a model's reasoning trace.\n"
        "Ground-truth reasons are separated by a new line symbol. You should first identy how many reasons are there.\n"
        "Then, for EACH ground-truth reason, in order, you MUST return exactly one classification object.\n"
        "Do NOT merge reasons. Do NOT skip any reason.\n\n"
        "A ground-truth reason is considered present if the same clinical or logical justification "
        "is explicitly stated or clearly paraphrased in the model's reasoning.\n\n"
        "Use only 'Yes' (1) or 'No' (0) as a binary classification. "
        "Output JSON with a brief explanation for each decision."
    )

    input_model = QCR
    output_model = ReasoningRecallClassifications

    examples = [
        (
            QCR(
                ground_truth_reasoning=(
                    "Lack of response to antibiotics argues against infection.\nHigh T2 signal with normal CT supports a soft-tissue nerve lesion.\nHistology showing disorganized nerve bundles is characteristic of traumatic neuroma.\nAbsence of systemic features makes syndromic mucosal neuroma unlikely."
                ),
                model_reasoning=(
                    "The lesion is painful and localized, suggesting a nerve-related process.\nMRI shows high T2 signal without bony involvement, consistent with a soft-tissue nerve lesion.\nBiopsy revealed disorganized nerve bundles, supporting traumatic neuroma."
                ),
            ),
            ReasoningRecallClassifications(
                classifications=[
                    ReasoningRecallClassification(
                        statement="Lack of response to antibiotics argues against infection.",
                        reason="The model reasoning does not mention antibiotic treatment or infection exclusion.",
                        attributed=0,
                    ),
                    ReasoningRecallClassification(
                        statement="High T2 signal with normal CT supports a soft-tissue nerve lesion.",
                        reason="The reasoning explicitly mentions high T2 signal and lack of bony involvement.",
                        attributed=1,
                    ),
                    ReasoningRecallClassification(
                        statement="Histology showing disorganized nerve bundles is characteristic of traumatic neuroma.",
                        reason="The reasoning explicitly refers to disorganized nerve bundles on biopsy.",
                        attributed=1,
                    ),
                    ReasoningRecallClassification(
                        statement="Absence of systemic features makes syndromic mucosal neuroma unlikely.",
                        reason="The reasoning does not discuss systemic or syndromic features.",
                        attributed=0,
                    ),
                ]
            ),
        ),
    ]


@dataclass
class ReasoningRecall(MetricWithLLM, SingleTurnMetric):
    """
    Reasoning Recall measures how many ground-truth reasoning steps from the
    case report are recovered in the model's reasoning trace.

    For a case i:
        Ri = set of ground-truth reasons
        Ti = reasons present in the model reasoning

        ci = |Ri ∩ Ti| / |Ri|

    The final score is the average of ci across cases.
    """

    name: str = "reasoning_recall"

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"reference", "response"},
        }
    )

    reasoning_recall_prompt: PydanticPrompt = ReasoningRecallClassificationPrompt()

    async def _ascore(self, row):
        # routing handled by SingleTurnMetric / MultiTurnMetric
        raise NotImplementedError

    async def _single_turn_ascore(self, sample, callbacks):
        prompt_input = QCR(
            ground_truth_reasoning=sample.reference,
            model_reasoning=sample.response,
        )

        prompt_response = await self.reasoning_recall_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks,
        )

        classifications = prompt_response.classifications

        if not classifications:
            return 0.0

        print(
            {
                "classifications": [
                    {
                        "statement": c.statement,
                        "reason": c.reason,
                        "attributed": int(c.attributed),
                    }
                    for c in classifications
                ]
            }
        )

        attributed = [c.attributed for c in classifications]
        return sum(attributed) / len(attributed)
