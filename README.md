# Getting started

## Prerequisite

- [ ] Docker Desktop
- [ ] Cohere
- [ ] LLM inference (Ollama/vLLM/huggingface) (VLM)

## Initialization

1. Install dependencies and activate

    ```bash
    uv sync
    ```

    > _conflicting dependencies --> remove conflicting library zhipou_

    ```bash
    source .venv/bin/activate
    ```

2. Setup `.env` from `.env.example`

3. Init LightRAG instance

    From repo root, run:

    ```bash
    docker compose up
    ```

> _Postgres DB not exists --> nuke everything (delete /data, docker compose down -v)_

> _Automatically pull image and build_

> _Access LightRAG instance locally via http://localhost:9621_

4. For if you want to test VLM doc input

With virtual environment activated, run

```bash
python src/test2.py
```

> _connect to LightRAG instance to see the import result._

## Notes

- Base url for huggingface: https://router.huggingface.co/v1
- Base url for vllm local host: http://localhost:8002/v1

## Deploy local model with vLLM

```bash
cd vllm-docker
```

or pull repo

```bash
git clone https://github.com/fuisl/vllm-test.git
```

Then

```bash
docker compose up
```
