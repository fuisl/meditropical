"""
RAGAS Evaluation Script

Evaluates RAG response quality using RAGAS metrics:
- Answer Relevance: Is the answer relevant to the question?
- Answer Accuracy: Is the answer accurate to the ground-truth?

Usage:
    # Specify dataset
    python evaluation/eval_json.py --dataset my_test.json
    python evaluation/eval_json.py -d my_test.json

    # Get help
    python evaluation/eval_json.py --help

Results are saved to: evaluation/results/
    - results_YYYYMMDD_HHMMSS.csv   (CSV export for analysis)
    - results_YYYYMMDD_HHMMSS.json  (Full results with details)

Technical Notes:
    - Uses stable RAGAS API (LangchainLLMWrapper) for maximum compatibility
    - Supports custom OpenAI-compatible endpoints via EVAL_LLM_BINDING_HOST
    - Enables bypass_n mode for endpoints that don't support 'n' parameter
    - Deprecation warnings are suppressed for cleaner output
"""

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from lightrag.utils import logger

# Suppress LangchainLLMWrapper deprecation warning
# We use LangchainLLMWrapper for stability and compatibility with all RAGAS versions
warnings.filterwarnings(
    "ignore",
    message=".*LangchainLLMWrapper is deprecated.*",
    category=DeprecationWarning,
)

# Suppress token usage warning for custom OpenAI-compatible endpoints
# Custom endpoints (vLLM, SGLang, etc.) often don't return usage information
# This is non-critical as token tracking is not required for RAGAS evaluation
warnings.filterwarnings(
    "ignore",
    message=".*Unexpected type for token usage.*",
    category=UserWarning,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# use the .env that is inside the current folder
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# Conditional imports - will raise ImportError if dependencies not installed
try:
    from datasets import Dataset
    from ragas import evaluate
    from evaluation.metrics.answer_accuracy import AnswerAccuracy
    from metrics.answer_relevance import AnswerRelevance as AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    from tqdm import tqdm

    IMPORT_SUCCESS = True

except ImportError:
    IMPORT_SUCCESS = False
    Dataset = None
    evaluate = None
    LangchainLLMWrapper = None


CONNECT_TIMEOUT_SECONDS = 60.0
READ_TIMEOUT_SECONDS = 120.0
TOTAL_TIMEOUT_SECONDS = 180.0


def _is_nan(value: Any) -> bool:
    """Return True when value is a float NaN."""
    return isinstance(value, float) and math.isnan(value)


class Evaluator:
    """Evaluate RAG system quality using RAGAS metrics"""

    def __init__(self, test_dataset_path: str = None, rag_api_url: str = None):
        """
        Initialize evaluator with test dataset

        Args:
            test_dataset_path: Path to test dataset JSON file

        Environment Variables:
            EVAL_LLM_MODEL: LLM model for evaluation (default: gpt-4o-mini)
            EVAL_EMBEDDING_MODEL: Embedding model for evaluation (default: text-embedding-3-small)
            EVAL_LLM_BINDING_API_KEY: API key for LLM (fallback to OPENAI_API_KEY)
            EVAL_LLM_BINDING_HOST: Custom endpoint URL for LLM (optional)
            EVAL_EMBEDDING_BINDING_API_KEY: API key for embeddings (fallback: EVAL_LLM_BINDING_API_KEY -> OPENAI_API_KEY)
            EVAL_EMBEDDING_BINDING_HOST: Custom endpoint URL for embeddings (fallback: EVAL_LLM_BINDING_HOST)

        Raises:
            ImportError: If ragas or datasets packages are not installed
            EnvironmentError: If EVAL_LLM_BINDING_API_KEY and OPENAI_API_KEY are both not set
        """
        # Validate RAGAS dependencies are installed
        if not IMPORT_SUCCESS:
            raise ImportError(
                "Import error. Please ensure you have installed ragas, datasets, langchain_ollama, etc."
            )

        eval_model = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
        eval_llm_base_url = os.getenv("EVAL_LLM_BINDING_HOST")

        # Configure evaluation embeddings (for RAGAS scoring)
        # Fallback chain: EVAL_EMBEDDING_BINDING_API_KEY -> EVAL_LLM_BINDING_API_KEY -> OPENAI_API_KEY
        eval_embedding_api_key = (
            os.getenv("EVAL_EMBEDDING_BINDING_API_KEY")
            or os.getenv("EVAL_LLM_BINDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        eval_embedding_model = os.getenv(
            "EVAL_EMBEDDING_MODEL", "text-embedding-3-large"
        )
        # Fallback chain: EVAL_EMBEDDING_BINDING_HOST -> EVAL_LLM_BINDING_HOST -> None
        eval_embedding_base_url = os.getenv("EVAL_EMBEDDING_BINDING_HOST") or os.getenv(
            "EVAL_LLM_BINDING_HOST"
        )

        # Create LLM and Embeddings instances for RAGAS
        llm_kwargs = {
            "model": eval_model,
            # "api_key": eval_llm_api_key,
            "max_retries": int(os.getenv("EVAL_LLM_MAX_RETRIES", "5")),
            "request_timeout": int(os.getenv("EVAL_LLM_TIMEOUT", "3600")),
        }
        embedding_kwargs = {
            "model": eval_embedding_model,
            "api_key": eval_embedding_api_key,
        }

        if eval_llm_base_url:
            llm_kwargs["base_url"] = eval_llm_base_url

        if eval_embedding_base_url:
            embedding_kwargs["base_url"] = eval_embedding_base_url

        base_llm = ChatOpenAI(**llm_kwargs)
        self.eval_embeddings = OpenAIEmbeddings(**embedding_kwargs)

        # Wrap LLM with LangchainLLMWrapper and enable bypass_n mode for custom endpoints
        # This ensures compatibility with endpoints that don't support the 'n' parameter
        # by generating multiple outputs through repeated prompts instead of using 'n' parameter
        try:
            self.eval_llm = LangchainLLMWrapper(
                langchain_llm=base_llm,
                bypass_n=True,  # Enable bypass_n to avoid passing 'n' to OpenAI API
            )
            logger.debug("Successfully configured bypass_n mode for LLM wrapper")
        except Exception as e:
            logger.warning(
                "Could not configure LangchainLLMWrapper with bypass_n: %s. "
                "Using base LLM directly, which may cause warnings with custom endpoints.",
                e,
            )
            self.eval_llm = base_llm

        assert test_dataset_path

        self.test_dataset_path = Path(test_dataset_path)
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Load test dataset
        self.test_cases = self._load_test_dataset()

        # Store configuration values for display
        self.eval_model = eval_model
        self.eval_embedding_model = eval_embedding_model
        self.eval_llm_base_url = eval_llm_base_url
        self.eval_embedding_base_url = eval_embedding_base_url
        self.eval_max_retries = llm_kwargs["max_retries"]
        self.eval_timeout = llm_kwargs["request_timeout"]

        # Display configuration
        self._display_configuration()

    def _display_configuration(self):
        """Display all evaluation configuration settings"""
        logger.info("Evaluation Models:")
        logger.info("  • LLM Model:            %s", self.eval_model)
        logger.info("  • Embedding Model:      %s", self.eval_embedding_model)

        # Display LLM endpoint
        if self.eval_llm_base_url:
            logger.info("  • LLM Endpoint:         %s", self.eval_llm_base_url)
            logger.info(
                "  • Bypass N-Parameter:   Enabled (use LangchainLLMWrapper for compatibility)"
            )
        else:
            logger.info("  • LLM Endpoint:         OpenAI Official API")

        # Display Embedding endpoint (only if different from LLM)
        if self.eval_embedding_base_url:
            if self.eval_embedding_base_url != self.eval_llm_base_url:
                logger.info(
                    "  • Embedding Endpoint:   %s", self.eval_embedding_base_url
                )
            # If same as LLM endpoint, no need to display separately
        elif not self.eval_llm_base_url:
            # Both using OpenAI - already displayed above
            pass
        else:
            # LLM uses custom endpoint, but embeddings use OpenAI
            logger.info("  • Embedding Endpoint:   OpenAI Official API")

        logger.info("Test Configuration:")
        logger.info("  • Total Test Cases:     %s", len(self.test_cases))
        logger.info("  • Test Dataset:         %s", self.test_dataset_path.name)
        logger.info("  • Results Directory:    %s", self.results_dir.name)

    def _load_test_dataset(self) -> List[Dict[str, str]]:
        """Load test cases from JSON file"""
        if not self.test_dataset_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {self.test_dataset_path}")

        with open(self.test_dataset_path) as f:
            data = json.load(f)

        return data.get("test_cases", [])

    async def evaluate_single_case(
        self,
        idx: int,
        test_case: Dict[str, str],
        rag_semaphore: asyncio.Semaphore,
        eval_semaphore: asyncio.Semaphore,
        client: httpx.AsyncClient,
        progress_counter: Dict[str, int],
        position_pool: asyncio.Queue,
        pbar_creation_lock: asyncio.Lock,
    ) -> Dict[str, Any]:
        """
        Evaluate a single test case with two-stage pipeline concurrency control

        Args:
            idx: Test case index (1-based)
            test_case: Test case dictionary with question and ground_truth
            rag_semaphore: Semaphore to control overall concurrency (covers entire function)
            eval_semaphore: Semaphore to control RAGAS evaluation concurrency (Stage 2)
            client: Shared httpx AsyncClient for connection pooling
            progress_counter: Shared dictionary for progress tracking
            position_pool: Queue of available tqdm position indices
            pbar_creation_lock: Lock to serialize tqdm creation and prevent race conditions

        Returns:
            Evaluation result dictionary
        """
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
        answer = test_case["answer"]
        contexts = test_case["contexts"]

        # Prepare dataset for RAGAS evaluation with CORRECT contexts
        eval_dataset = Dataset.from_dict(
            {
                "user_input": [question],
                "answer": [answer],
                "retrieved_contexts": [contexts],
                "ground_truth": [ground_truth],
            }
        )

        # Stage 2: Run RAGAS evaluation (controlled by eval_semaphore)
        # IMPORTANT: Create fresh metric instances for each evaluation to avoid
        # concurrent state conflicts when multiple tasks run in parallel
        async with eval_semaphore:
            pbar = None
            position = None
            try:
                # Acquire a position from the pool for this tqdm progress bar
                position = await position_pool.get()

                # Serialize tqdm creation to prevent race conditions
                # Multiple tasks creating tqdm simultaneously can cause display conflicts
                async with pbar_creation_lock:
                    # Create tqdm progress bar with assigned position to avoid overlapping
                    # leave=False ensures the progress bar is cleared after completion,
                    # preventing accumulation of completed bars and allowing position reuse
                    pbar = tqdm(
                        total=3,
                        desc=f"Eval-{idx:02d}",
                        position=position,
                        leave=False,
                    )
                    # Give tqdm time to initialize and claim its screen position
                    await asyncio.sleep(0.05)

                eval_results = evaluate(
                    dataset=eval_dataset,
                    metrics=[AnswerRelevancy(), AnswerAccuracy()],
                    llm=self.eval_llm,
                    embeddings=self.eval_embeddings,
                    _pbar=pbar,
                )

                # Convert to DataFrame (RAGAS v0.3+ API)
                df = eval_results.to_pandas()

                # Extract scores from first row
                scores_row = df.iloc[0]

                # Extract scores (RAGAS v0.3+ uses .to_pandas())
                result = {
                    "test_number": idx,
                    "question": question,
                    "answer": test_case["answer"],
                    "ground_truth": ground_truth,
                    "contexts": contexts,
                    "project": test_case.get("project", "unknown"),
                    "metrics": {
                        "answer_relevance": float(
                            scores_row.get("answer_relevance", 0)
                        ),
                        "answer_accuracy": float(scores_row.get("answer_accuracy", 0)),
                    },
                    "timestamp": datetime.now().isoformat(),
                }

                # Calculate RAGAS score (average of all metrics, excluding NaN values)
                metrics = result["metrics"]
                valid_metrics = [v for v in metrics.values() if not _is_nan(v)]
                ragas_score = (
                    sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0
                )
                result["ragas_score"] = round(ragas_score, 4)

                # Update progress counter
                progress_counter["completed"] += 1

                return result

            except Exception as e:
                logger.error("Error evaluating test %s: %s", idx, str(e))
                progress_counter["completed"] += 1
                return {
                    "test_number": idx,
                    "question": question,
                    "error": str(e),
                    "metrics": {},
                    "ragas_score": 0,
                    "timestamp": datetime.now().isoformat(),
                }
            finally:
                # Force close progress bar to ensure completion
                if pbar is not None:
                    pbar.close()
                # Release the position back to the pool for reuse
                if position is not None:
                    await position_pool.put(position)

    async def evaluate_responses(self) -> List[Dict[str, Any]]:
        """
        Evaluate all test cases in parallel with two-stage pipeline and return metrics

        Returns:
            List of evaluation results with metrics
        """
        # Get evaluation concurrency from environment (default to 2 for parallel evaluation)
        max_async = int(os.getenv("EVAL_MAX_CONCURRENT", "2"))

        logger.info("%s", "=" * 70)
        logger.info("🚀 Starting RAGAS Evaluation")
        logger.info("🔧 RAGAS Evaluation (Stage 2): %s concurrent", max_async)
        logger.info("%s", "=" * 70)

        # Create two-stage pipeline semaphores
        # Stage 1: RAG generation - allow x2 concurrency to keep evaluation fed
        rag_semaphore = asyncio.Semaphore(max_async * 2)
        # Stage 2: RAGAS evaluation - primary bottleneck
        eval_semaphore = asyncio.Semaphore(max_async)

        # Create progress counter (shared across all tasks)
        progress_counter = {"completed": 0}

        # Create position pool for tqdm progress bars
        # Positions range from 0 to max_async-1, ensuring no overlapping displays
        position_pool = asyncio.Queue()
        for i in range(max_async):
            await position_pool.put(i)

        # Create lock to serialize tqdm creation and prevent race conditions
        # This ensures progress bars are created one at a time, avoiding display conflicts
        pbar_creation_lock = asyncio.Lock()

        # Create shared HTTP client with connection pooling and proper timeouts
        # Timeout: 3 minutes for connect, 5 minutes for read (LLM can be slow)
        timeout = httpx.Timeout(
            TOTAL_TIMEOUT_SECONDS,
            connect=CONNECT_TIMEOUT_SECONDS,
            read=READ_TIMEOUT_SECONDS,
        )
        limits = httpx.Limits(
            max_connections=(max_async + 1) * 2,  # Allow buffer for RAG stage
            max_keepalive_connections=max_async + 1,
        )

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            # Create tasks for all test cases
            tasks = [
                self.evaluate_single_case(
                    idx,
                    test_case,
                    rag_semaphore,
                    eval_semaphore,
                    client,
                    progress_counter,
                    position_pool,
                    pbar_creation_lock,
                )
                for idx, test_case in enumerate(self.test_cases, 1)
            ]

            # Run all evaluations in parallel (limited by two-stage semaphores)
            results = await asyncio.gather(*tasks)

        return list(results)

    def _export_to_csv(self, results: List[Dict[str, Any]]) -> Path:
        """
        Export evaluation results to CSV file

        Args:
            results: List of evaluation results

        Returns:
            Path to the CSV file

        CSV Format:
            - question: The test question
            - project: Project context
            - answer_relevance: Answer relevance score (0-1)
            - answer_accuracy: Answer accuracy (0 or 1)
            - ragas_score: Overall RAGAS score (0-1)
            - timestamp: When evaluation was run
        """
        dataset_name = self.test_dataset_path.stem
        csv_path = (
            self.results_dir / f"eval_answer_{dataset_name}.csv"
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "test_number",
                "question",
                "project",
                "answer_relevance",
                "answer_accuracy",
                "ragas_score",
                "status",
                "timestamp",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx, result in enumerate(results, 1):
                metrics = result.get("metrics", {})
                writer.writerow(
                    {
                        "test_number": idx,
                        "question": result.get("question", ""),
                        "project": result.get("project", "unknown"),
                        "answer_relevance": f"{metrics.get('answer_relevance', 0):.4f}",
                        "answer_accuracy": f"{metrics.get('answer_accuracy', 0):.4f}",
                        "ragas_score": f"{result.get('ragas_score', 0):.4f}",
                        "status": "success" if metrics else "error",
                        "timestamp": result.get("timestamp", ""),
                    }
                )

        return csv_path

    def _format_metric(self, value: float, width: int = 6) -> str:
        """
        Format a metric value for display, handling NaN gracefully

        Args:
            value: The metric value to format
            width: The width of the formatted string

        Returns:
            Formatted string (e.g., "0.8523" or "  N/A ")
        """
        if _is_nan(value):
            return "N/A".center(width)
        return f"{value:.4f}".rjust(width)

    def _display_results_table(self, results: List[Dict[str, Any]]):
        """
        Display evaluation results in a formatted table

        Args:
            results: List of evaluation results
        """
        logger.info("")
        logger.info("%s", "=" * 115)
        logger.info("📊 EVALUATION RESULTS SUMMARY")
        logger.info("%s", "=" * 115)

        # Table header
        logger.info(
            "%-4s | %-50s | %6s | %7s | %6s |%6s | %6s",
            "#",
            "Question",
            "Faith",
            "AnswRel",
            "AnswAcc",
            "RAGAS",
            "Status",
        )
        logger.info("%s", "-" * 115)

        # Table rows
        for result in results:
            test_num = result.get("test_number", 0)
            question = result.get("question", "")
            # Truncate question to 50 chars
            question_display = (
                (question[:47] + "...") if len(question) > 50 else question
            )

            metrics = result.get("metrics", {})
            if metrics:
                # Success case - format each metric, handling NaN values
                ans_rel = metrics.get("answer_relevance", 0)
                ans_acc = metrics.get("answer_accuracy", 0)
                ragas = result.get("ragas_score", 0)
                status = "✓"

                logger.info(
                    "%-4d | %-50s | %s | %s | %s | %s | %6s",
                    test_num,
                    question_display,
                    self._format_metric(ans_rel, 7),
                    self._format_metric(ans_acc, 6),
                    self._format_metric(ragas, 6),
                    status,
                )
            else:
                # Error case
                error = result.get("error", "Unknown error")
                error_display = (error[:20] + "...") if len(error) > 23 else error
                logger.info(
                    "%-4d | %-50s | %6s | %7s | %6s | %6s | ✗ %s",
                    test_num,
                    question_display,
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    error_display,
                )

        logger.info("%s", "=" * 115)

    def _calculate_benchmark_stats(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate benchmark statistics from evaluation results

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with benchmark statistics
        """
        # Filter out results with errors
        valid_results = [r for r in results if r.get("metrics")]
        total_tests = len(results)
        successful_tests = len(valid_results)
        failed_tests = total_tests - successful_tests

        if not valid_results:
            return {
                "total_tests": total_tests,
                "successful_tests": 0,
                "failed_tests": failed_tests,
                "success_rate": 0.0,
            }

        # Calculate averages for each metric (handling NaN values correctly)
        # Track both sum and count for each metric to handle NaN values properly
        metrics_data = {
            "answer_relevance": {"sum": 0.0, "count": 0},
            "answer_accuracy": {"sum": 0.0, "count": 0},
            "ragas_score": {"sum": 0.0, "count": 0},
        }

        for result in valid_results:
            metrics = result.get("metrics", {})

            # For each metric, sum non-NaN values and count them
            answer_relevance = metrics.get("answer_relevance", 0)
            if not _is_nan(answer_relevance):
                metrics_data["answer_relevance"]["sum"] += answer_relevance
                metrics_data["answer_relevance"]["count"] += 1

            answer_accuracy = metrics.get("answer_accuracy", 0)
            if not _is_nan(answer_accuracy):
                metrics_data["answer_accuracy"]["sum"] += answer_accuracy
                metrics_data["answer_accuracy"]["count"] += 1

            ragas_score = result.get("ragas_score", 0)
            if not _is_nan(ragas_score):
                metrics_data["ragas_score"]["sum"] += ragas_score
                metrics_data["ragas_score"]["count"] += 1

        # Calculate averages using actual counts for each metric
        avg_metrics = {}
        for metric_name, data in metrics_data.items():
            if data["count"] > 0:
                avg_val = data["sum"] / data["count"]
                avg_metrics[metric_name] = (
                    round(avg_val, 4) if not _is_nan(avg_val) else 0.0
                )
            else:
                avg_metrics[metric_name] = 0.0

        # Find min and max RAGAS scores (filter out NaN)
        ragas_scores = []
        for r in valid_results:
            score = r.get("ragas_score", 0)
            if _is_nan(score):
                continue  # Skip NaN values
            ragas_scores.append(score)

        min_score = min(ragas_scores) if ragas_scores else 0
        max_score = max(ragas_scores) if ragas_scores else 0

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": round(successful_tests / total_tests * 100, 2),
            "average_metrics": avg_metrics,
            "min_ragas_score": round(min_score, 4),
            "max_ragas_score": round(max_score, 4),
        }

    async def run(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""

        start_time = time.time()

        # Evaluate responses
        results = await self.evaluate_responses()

        elapsed_time = time.time() - start_time

        # Calculate benchmark statistics
        benchmark_stats = self._calculate_benchmark_stats(results)

        # Save results
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "benchmark_stats": benchmark_stats,
            "results": results,
        }

        # Display results table
        self._display_results_table(results)

        # Save JSON results
        dataset_name = self.test_dataset_path.stem
        json_path = (
            self.results_dir
            / f"eval_answer_{dataset_name}.json"
        )
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Export to CSV
        csv_path = self._export_to_csv(results)

        # Print summary
        logger.info("")
        logger.info("%s", "=" * 70)
        logger.info("📊 EVALUATION COMPLETE")
        logger.info("%s", "=" * 70)
        logger.info("Total Tests:    %s", len(results))
        logger.info("Successful:     %s", benchmark_stats["successful_tests"])
        logger.info("Failed:         %s", benchmark_stats["failed_tests"])
        logger.info("Success Rate:   %.2f%%", benchmark_stats["success_rate"])
        logger.info("Elapsed Time:   %.2f seconds", elapsed_time)
        logger.info("Avg Time/Test:  %.2f seconds", elapsed_time / len(results))

        # Print benchmark metrics
        logger.info("")
        logger.info("%s", "=" * 70)
        logger.info("📈 BENCHMARK RESULTS (Average)")
        logger.info("%s", "=" * 70)
        avg = benchmark_stats["average_metrics"]
        logger.info("Average Answer Relevance:  %.4f", avg["answer_relevance"])
        logger.info("Average Answer Accuracy:    %.4f", avg["answer_accuracy"])
        logger.info("Average RAGAS Score:       %.4f", avg["ragas_score"])
        logger.info("%s", "-" * 70)
        logger.info(
            "Min RAGAS Score:           %.4f",
            benchmark_stats["min_ragas_score"],
        )
        logger.info(
            "Max RAGAS Score:           %.4f",
            benchmark_stats["max_ragas_score"],
        )

        logger.info("")
        logger.info("%s", "=" * 70)
        logger.info("📁 GENERATED FILES")
        logger.info("%s", "=" * 70)
        logger.info("Results Dir:    %s", self.results_dir.absolute())
        logger.info("   • CSV:  %s", csv_path.name)
        logger.info("   • JSON: %s", json_path.name)
        logger.info("%s", "=" * 70)

        return summary


async def main():
    """
    Main entry point for RAGAS evaluation

    Command-line arguments:
        --dataset, -d: Path to test dataset JSON file (default: sample_dataset.json)

    Usage:
        python evaluation/eval_rag_quality.py --dataset my_test.json
    """
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(
            description="RAGAS Evaluation Script",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:

    # Specify dataset
    python evaluation/eval_rag_quality.py --dataset my_test.json
            """,
        )

        parser.add_argument(
            "--dataset",
            "-d",
            type=str,
            default=None,
            help="Path to test dataset JSON file (default: sample_dataset.json in evaluation directory)",
        )

        args = parser.parse_args()

        logger.info("%s", "=" * 70)
        logger.info("🔍 RAGAS Evaluation")
        logger.info("%s", "=" * 70)

        evaluator = Evaluator(test_dataset_path=args.dataset)
        await evaluator.run()
    except Exception as e:
        logger.exception("❌ Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
