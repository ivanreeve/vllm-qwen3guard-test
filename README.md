# pii-detection-test

Lightweight test harness for evaluating PII detection behavior of a Qwen3Guard model served through Ollama.

## What is in this repo

- `detect_pii.py`: Runs the evaluation against a JSON dataset and prints metrics.
- `data/pii_test_dataset.json`: Test dataset used by default.
- `pyproject.toml` / `uv.lock`: Project and dependency definitions for `uv`.

## Prerequisites

- Docker + Docker Compose v2 (for container workflow)
- Python `3.13+` + [`uv`](https://docs.astral.sh/uv/) (for local non-Docker workflow)

## Quick Setup (Docker)

For a first-time run using Docker only:

```bash
git clone <repo-url>
cd pii-detection-test
mkdir -p results
docker compose up -d --build
docker compose logs -f ollama-pull evaluator
ls -lh results/results.json
```

## Local Setup (Non-Docker)

1. Install dependencies:

```bash
uv sync
```

2. Start Ollama in a separate terminal:

```bash
ollama serve
```

3. Pull the model:

```bash
ollama pull sileader/qwen3guard:0.6b
```

## Local Run

Default run:

```bash
uv run python detect_pii.py
```

Verbose run:

```bash
uv run python detect_pii.py --verbose
```

Save full JSON results:

```bash
uv run python detect_pii.py --output results.json
```

## Docker Setup (Ollama + Evaluator in Docker)

This is the default container workflow and runs both Ollama and the evaluator in Docker.

1. Ensure a results folder exists:

```bash
mkdir -p results
```

2. Start the full stack:

```bash
docker compose up -d --build
```

What happens:
- `ollama` starts in Docker.
- `ollama-pull` pulls model `sileader/qwen3guard:0.6b`.
- `evaluator` runs `detect_pii.py` against `http://ollama:11434` and writes `results/results.json`.

3. Watch progress:

```bash
docker compose logs -f ollama-pull evaluator
```

4. Verify output file:

```bash
ls -lh results/results.json
```

Run everything from a single Python command (start Ollama container, pull model, run evaluator, print summary):

```bash
python3 run_docker_eval.py --verbose
```

Re-run evaluator only (without restarting ollama):

```bash
docker compose run --rm evaluator
```

Use a different model:

```bash
OLLAMA_MODEL=your-org/your-model:tag docker compose up -d --build
```

Stop services:

```bash
docker compose down
```

## Cleanup

Stop containers only (keep downloaded model data for next run):

```bash
docker compose down
```

Full cleanup for this project (remove containers + Ollama model volume + evaluator image):

```bash
docker compose down -v --remove-orphans
docker image rm pii-detection-test:latest
```

Optional: also remove the base Ollama image:

```bash
docker image rm ollama/ollama:latest
```

Optional: remove generated results:

```bash
rm -rf results
```

## CLI options

```text
--model MODEL              Ollama model name (default: sileader/qwen3guard:0.6b)
--ollama-url OLLAMA_URL    Ollama server URL (default: http://localhost:11434)
--dataset DATASET          Path to test dataset JSON
--output OUTPUT            Save full results to JSON file
--verbose                  Show per-query raw model output
```

## Notes

- The evaluator treats a case as PII only when model output is `Safety: Unsafe` or `Safety: Controversial` and categories include `PII`.
- If Ollama is unreachable, verify the URL and that Ollama is running.
- If the model name differs locally, pass it via `--model`.
