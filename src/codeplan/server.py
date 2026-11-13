"""Minimal local inference server (FastAPI + uvicorn).

Run::

    codeplan-serve --checkpoint checkpoints/sft_50m/last.pt --host 0.0.0.0 --port 8000

Then ``POST /generate`` with ``{"prompt": "..."}``.
Requires the ``serve`` extra: ``pip install 'codeplan-lm[serve]'``.
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from typing import Any

import torch

from .generate import load_model_from_checkpoint
from .model import generate
from .tokenizer import DEFAULT_SYSTEM_PROMPT

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>CodePlan-LM</title></head>
<body style="font-family:system-ui;max-width:44rem;margin:3rem auto">
<h1>CodePlan-LM</h1>
<textarea id="prompt" rows="5" style="width:100%">Write a Python function add(a, b)</textarea><br><br>
<button id="go">Generate</button>
<pre id="out" style="white-space:pre-wrap"></pre>
<p>API: <code>POST /generate {"prompt": "..."}</code></p>
<script>
document.getElementById("go").onclick = async () => {
  const out = document.getElementById("out");
  out.textContent = "\\u2026";
  const response = await fetch("/generate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({prompt: document.getElementById("prompt").value, max_new_tokens: 128}),
  });
  const data = await response.json();
  out.textContent = data.text ?? JSON.stringify(data);
};
</script>
</body></html>
"""


@dataclass
class GenerateOptions:
    prompt: str
    system: str = DEFAULT_SYSTEM_PROMPT
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.05

    @classmethod
    def from_payload(cls, payload: dict) -> "GenerateOptions":
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("field 'prompt' is required")
        options = cls(
            prompt=prompt,
            system=str(payload.get("system") or DEFAULT_SYSTEM_PROMPT),
            max_new_tokens=int(payload.get("max_new_tokens", 256)),
            temperature=float(payload.get("temperature", 0.7)),
            top_k=int(payload.get("top_k", 50)),
            top_p=float(payload.get("top_p", 0.95)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.05)),
        )
        if not 1 <= options.max_new_tokens <= 2048:
            raise ValueError("max_new_tokens must be between 1 and 2048")
        if not 0.0 <= options.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not 0.0 < options.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if options.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        return options


def create_app(checkpoint: str, tokenizer_path: str | None = None, device: str = "auto"):
    from fastapi import Body, FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    model, tokenizer, _ = load_model_from_checkpoint(checkpoint, tokenizer_path, device)
    model_device = next(model.parameters()).device
    lock = threading.Lock()

    def run_generation(options: GenerateOptions) -> dict[str, Any]:
        prompt = tokenizer.format_chat(options.prompt, options.system)
        ids = tokenizer.encode(prompt)
        keep = max(1, model.cfg.max_seq_len - options.max_new_tokens)
        if len(ids) > keep:
            ids = ids[-keep:]
        input_ids = torch.tensor([ids], dtype=torch.long, device=model_device)
        with lock:
            out = generate(
                model,
                input_ids,
                max_new_tokens=options.max_new_tokens,
                temperature=options.temperature,
                top_k=options.top_k,
                top_p=options.top_p,
                repetition_penalty=options.repetition_penalty,
                eos_token_id=tokenizer.eos_id,
            )
        new_tokens = out[0, input_ids.shape[1]:].tolist()
        return {
            "text": tokenizer.decode(new_tokens, skip_special=True).strip(),
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": len(new_tokens),
        }

    app = FastAPI(title="CodePlan-LM", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "parameters": model.num_parameters(),
            "vocab_size": tokenizer.vocab_size,
            "device": str(model_device),
            "context": model.cfg.max_seq_len,
        }

    @app.post("/generate")
    def generate_endpoint(payload: dict = Body(...)) -> Any:
        try:
            options = GenerateOptions.from_payload(payload)
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return run_generation(options)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve CodePlan-LM via HTTP.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("uvicorn is required: pip install 'codeplan-lm[serve]'") from exc

    app = create_app(args.checkpoint, args.tokenizer, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
