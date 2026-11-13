"""Small coding benchmark with unit-test execution.

Each task is one JSONL record::

    {"id": "add", "prompt": "Write add(a, b).", "tests": "assert add(2, 3) == 5"}

The model answer is generated, the ``<code>`` block is extracted and the tests
are executed in a short-lived subprocess with a timeout and resource limits.

WARNING: this is *not* a real sandbox. It limits CPU time, address space and
wall-clock time, but generated code still runs as your user. Only use it on
code you are willing to execute; run it in a container/VM for untrusted models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from .generate import load_model_from_checkpoint
from .model import generate
from .prepare import detect_lang

CODE_BLOCK = re.compile(r"<code>(.*?)</code>", re.DOTALL)
FENCE_BLOCK = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)

DEFAULT_SYSTEM = (
    "You are CodePlan-LM, a coding assistant. Answer in German or English. "
    "Put runnable Python code inside <code>...</code>."
)


def extract_code(answer: str) -> str:
    """Pick the executable part of a model answer."""
    blocks = CODE_BLOCK.findall(answer)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)
    fences = FENCE_BLOCK.findall(answer)
    if fences:
        return "\n\n".join(block.strip() for block in fences)
    return answer.strip()


def _limits(timeout: int):
    """Best-effort POSIX resource limits for the child process."""

    def apply() -> None:  # pragma: no cover - runs in the child
        import resource

        cpu = max(1, int(timeout))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        memory = 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))

    return apply if os.name == "posix" else None


def run_tests(code: str, tests: str, timeout: int = 10) -> tuple[bool, str]:
    """Execute ``code`` followed by ``tests``. Returns ``(passed, output)``."""
    if not code.strip():
        return False, "empty code"
    script = f"{code}\n\n\n{tests}\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "answer.py"
        path.write_text(script, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                preexec_fn=_limits(timeout),
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
        output = (result.stdout + result.stderr).strip()
        if result.returncode < 0:
            output = f"terminated by signal {-result.returncode}\n{output}".strip()
        return result.returncode == 0, output[-2000:]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_tasks(path: str | Path) -> list[dict]:
    tasks = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            if not task.get("prompt") or not task.get("tests"):
                raise ValueError(f"Task needs 'prompt' and 'tests': {task}")
            task.setdefault("id", f"task-{len(tasks):04d}")
            tasks.append(task)
    if not tasks:
        raise ValueError(f"No tasks in {path}")
    return tasks


def evaluate(
    checkpoint: str,
    tasks_path: str,
    tokenizer_path: str | None = None,
    k: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    device: str = "auto",
    timeout: int = 10,
    limit: int = 0,
    allow_exec: bool = False,
) -> dict:
    if not allow_exec:
        raise SystemExit(
            "Refusing to execute model output. Re-run with --allow-exec "
            "(this runs generated code as your user; use a container for real evals)."
        )

    model, tokenizer, _ = load_model_from_checkpoint(checkpoint, tokenizer_path, device)
    device_t = next(model.parameters()).device
    tasks = load_tasks(tasks_path)
    if limit > 0:
        tasks = tasks[:limit]

    results = []
    passed_total = 0
    langs: dict[str, int] = {}
    for task in tasks:
        prompt = tokenizer.format_chat(task["prompt"], DEFAULT_SYSTEM)
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device_t)
        outcomes = []
        for _ in range(k):
            out = generate(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token_id=tokenizer.eos_id,
            )
            answer = tokenizer.decode(out[0, input_ids.shape[1]:].tolist(), skip_special=True)
            code = extract_code(answer)
            ok, output = run_tests(code, task["tests"], timeout=timeout)
            lang = detect_lang(answer)
            langs[lang] = langs.get(lang, 0) + 1
            outcomes.append({"passed": ok, "output": output, "code": code})
        c = sum(1 for o in outcomes if o["passed"])
        passed_total += c
        score = pass_at_k(k, c, k)
        results.append(
            {
                "id": task["id"],
                "prompt": task["prompt"],
                "passed_samples": c,
                "k": k,
                "pass_at_k": score,
                "samples": outcomes,
            }
        )
        print(f"{task['id']:<24} pass@{k}={score:.2f}  ({c}/{k})")

    n = len(results)
    language_ok = sum(v for key, v in langs.items() if key in {"de", "en"})
    return {
        "checkpoint": str(checkpoint),
        "tasks": n,
        "k": k,
        "pass_at_1": passed_total / max(1, n * k),
        "mean_pass_at_k": sum(r["pass_at_k"] for r in results) / max(1, n),
        "language_compliance": language_ok / max(1, sum(langs.values())),
        "languages": langs,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run coding tasks with unit-test execution.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", required=True, help="JSONL with prompt + tests")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-exec", action="store_true", help="Required: runs generated code")
    parser.add_argument("--out", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    report = evaluate(
        args.checkpoint,
        args.tasks,
        args.tokenizer,
        args.k,
        args.max_new_tokens,
        args.temperature,
        args.device,
        args.timeout,
        args.limit,
        args.allow_exec,
    )
    summary = {key: report[key] for key in ("tasks", "k", "pass_at_1", "mean_pass_at_k",
                                            "language_compliance", "languages")}
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
