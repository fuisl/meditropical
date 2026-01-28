import os
import dotenv
import asyncio
from typing import Any, Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

from diagnosis_agent.state import GraphState, EvaluationResult

dotenv.load_dotenv()


class RAGASEvaluator:
    """Lightweight RAGAS evaluator for single-sample execution (SYNC)"""

    def __init__(
        self,
        eval_llm_model: str = "gemma3:latest",
        eval_embedding_model: str = "bge-m3:latest",
        eval_llm_base_url: str | None = None,
        eval_embedding_base_url: str | None = None,
    ):
        # --- LLM ---
        base_llm = ChatOllama(
            model=eval_llm_model,
            temperature=0.0,
            base_url=eval_llm_base_url,
            client_kwargs={"timeout": 180.0},
        )

        self.llm = LangchainLLMWrapper(
            langchain_llm=base_llm,
            bypass_n=True,
        )

        # --- Embeddings ---
        self.embeddings = OllamaEmbeddings(
            model=eval_embedding_model,
            base_url=eval_embedding_base_url,
            client_kwargs={"timeout": 120.0},
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> List[EvaluationResult]:
        """Evaluate a single QA pair (SYNC)"""

        dataset = Dataset.from_dict(
            {
                "user_input": [question],
                "answer": [answer],
                "retrieved_contexts": [contexts],
            }
        )

        # IMPORTANT: create metrics per call (RAGAS metrics are stateful)
        metrics = [
            AnswerRelevancy(),
            Faithfulness(),
        ]

        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        row = results.to_pandas().iloc[0]

        return [
            {
                "metric": "answer_relevancy",
                "score": float(row.get("answer_relevancy", 0.0)),
            },
            {
                "metric": "faithfulness",
                "score": float(row.get("faithfulness", 0.0)),
            },
        ]


# ---------- Graph Node ----------

eval_model = os.getenv("EVAL_LLM_MODEL", "gemma3:latest")
eval_llm_base_url = os.getenv("EVAL_LLM_BINDING_HOST")
eval_embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "bge-m3:latest")
eval_embedding_base_url = os.getenv("EVAL_EMBEDDING_BINDING_HOST")


async def evaluate_diagnosis(state: GraphState) -> Dict[str, Any]:
    evaluator = RAGASEvaluator(
        eval_llm_model=eval_model,
        eval_embedding_model=eval_embedding_model,
        eval_llm_base_url=eval_llm_base_url,
        eval_embedding_base_url=eval_embedding_base_url,
    )

    metrics = await asyncio.to_thread(
        evaluator.evaluate,
        state["question"],
        state["answer"],
        state["contexts"],
    )

    return {"evaluation": metrics}
