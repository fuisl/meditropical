# Meditropical

A medical diagnosis agent system powered by LightRAG and LangGraph for tropical disease diagnosis.

## Repository Structure

### Core Directories

- **`src/`** - Main source code
  - `agent/` - LangGraph agent implementation
  - `diagnosis_agent/` - Diagnosis agent logic
  - `data/` - Data processing utilities

- **`evaluation/`** - Evaluation scripts and metrics
  - `eval_answer.py` - Answer quality evaluation
  - `eval_answer_original.py` - Original answer evaluation
  - `eval_rag.py` - RAG retrieval evaluation
  - `eval_reasoning.py` - Reasoning evaluation
  - `combine_metrics.py` - Metrics aggregation
  - `populate_contexts.py` - Context population for evaluation
  - `sample_dataset.json` - Sample evaluation dataset
  - `generated_answers/` - Generated answer outputs
  - `metrics/` - Evaluation metrics results
  - `results/` - Evaluation results

- **`demo/`** - Demo data and documentation
  - `data/` - Demo datasets

- **`docs/`** - Project documentation
  - `SELFHOST.md` - Self-hosting guide

### Configuration Files

- **`docker-compose.yml`** - Docker services configuration (LightRAG, PostgreSQL, Qdrant, Redis)
- **`langgraph.json`** - LangGraph configuration
- **`pyproject.toml`** - Python project dependencies and configuration
- **`.env.example`** - Environment variables template
- **`config.ini/`** - Application configuration

### Submodules

- **`LightRAG/`** - LightRAG framework (forked from HKUDS/LightRAG)
- **`vllm-docker/`** - vLLM deployment setup
- **`medgenerate/`** - Medical case generation tools
- **`multimodal-project-report/`** - Project documentation and reports

## Services

The Docker Compose setup includes:
- **LightRAG API** - RAG service (port 9621)
- **PostgreSQL** - Vector database with pgvector (port 5432)
- **Qdrant** - Vector search engine (port 6333)
- **Redis** - Caching layer (port 6379)
