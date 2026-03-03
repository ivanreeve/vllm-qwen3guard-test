#!/usr/bin/env python3
"""Run Dockerized Ollama + evaluator and print a concise results summary."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "sileader/qwen3guard:0.6b"
DEFAULT_OUTPUT = "results/results.json"


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n$ {printable}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start Ollama in Docker, pull model, run evaluator in Docker, "
            "and print results summary"
        )
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to repo root containing docker-compose.yml",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Relative output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to detect_pii.py",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip docker compose build evaluator",
    )
    parser.add_argument(
        "--down",
        action="store_true",
        help="Run docker compose down after evaluation",
    )
    return parser.parse_args()


def print_summary(output_path: Path) -> None:
    if not output_path.exists():
        print(f"\nExpected output not found: {output_path}", file=sys.stderr)
        return

    with output_path.open() as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {})
    cm = metrics.get("confusion_matrix", {})

    print("\n================ Docker Evaluation Summary ================")
    print(f"Output File : {output_path}")
    print(f"Model       : {payload.get('model', 'N/A')}")
    print(f"Total       : {payload.get('total_entries', 'N/A')}")
    print(f"Accuracy    : {metrics.get('accuracy', 0):.3f}")
    print(f"Precision   : {metrics.get('precision', 0):.3f}")
    print(f"Recall      : {metrics.get('recall', 0):.3f}")
    print(f"F1          : {metrics.get('f1', 0):.3f}")
    print(
        "Confusion   : "
        f"TP={cm.get('tp', 0)} FP={cm.get('fp', 0)} "
        f"FN={cm.get('fn', 0)} TN={cm.get('tn', 0)}"
    )
    print("==========================================================")


def main() -> int:
    args = parse_args()
    project_dir = args.project_dir.resolve()

    compose_file = project_dir / "docker-compose.yml"
    if not compose_file.exists():
        print(
            f"docker-compose.yml not found in {project_dir}",
            file=sys.stderr,
        )
        return 2

    output_path = (project_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OLLAMA_MODEL"] = args.model

    try:
        if not args.skip_build:
            run_command(["docker", "compose", "build", "evaluator"], project_dir, env)

        run_command(["docker", "compose", "up", "-d", "ollama"], project_dir, env)
        run_command(["docker", "compose", "run", "--rm", "ollama-pull"], project_dir, env)

        evaluator_cmd = [
            "docker",
            "compose",
            "run",
            "--rm",
            "evaluator",
            "--ollama-url",
            "http://ollama:11434",
            "--model",
            args.model,
            "--output",
            args.output,
        ]
        if args.verbose:
            evaluator_cmd.append("--verbose")

        run_command(evaluator_cmd, project_dir, env)
        print_summary(output_path)

    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    finally:
        if args.down:
            try:
                run_command(["docker", "compose", "down"], project_dir, env)
            except subprocess.CalledProcessError:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
