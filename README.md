# qwen3guard-pii-test

Lightweight evaluation harness for testing Qwen3Guard PII detection through a
vLLM OpenAI-compatible API endpoint.

## What is in this repo

- `detect_pii.py`: Runs the evaluation against a JSON dataset and prints metrics.
- `data/pii_test_dataset.json`: Default evaluation dataset.
- `results/`: Output directory for JSON reports.

## Google Colab Quick Start

1. Enable a GPU runtime in Colab.
2. Clone this repo and install dependencies:

```bash
!git clone <repo-url>
%cd ollama-qwen3guard-test
!pip install -U vllm requests tabulate tqdm
```

3. Start vLLM server in the background:

```bash
!python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3Guard-Gen-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto > vllm.log 2>&1 &
!sleep 25
```

4. Run evaluation:

```bash
!mkdir -p results
!python detect_pii.py \
  --api-base http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3Guard-Gen-4B \
  --output results/results.json
```

5. Optional verbose run:

```bash
!python detect_pii.py \
  --api-base http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3Guard-Gen-4B \
  --output results/results.json \
  --verbose
```

## Local Run (Non-Colab)

Install dependencies:

```bash
pip install -U vllm requests tabulate tqdm
```

Start vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3Guard-Gen-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
```

Run evaluator:

```bash
python detect_pii.py --api-base http://localhost:8000/v1
```

## CLI options

```text
--model MODEL              Model name exposed by vLLM (default: Qwen/Qwen3Guard-Gen-4B)
--api-base API_BASE        OpenAI-compatible API base URL (default: http://localhost:8000/v1)
--api-key API_KEY          Optional API key (or set VLLM_API_KEY)
--timeout TIMEOUT          Per-request timeout in seconds (default: 120)
--dataset DATASET          Path to test dataset JSON
--output OUTPUT            Save full results to JSON file
--verbose                  Show per-query raw model output
```

## Notes

- The evaluator expects model responses in the Qwen3Guard format:
  `Safety: ...`, `Categories: ...`, `Refusal: ...`.
- A case is counted as PII only when:
  `Safety` is `Unsafe` or `Controversial` and categories include `PII`.
- If the server is unreachable, verify vLLM is running and that `--api-base` is correct.
