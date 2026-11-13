"""HTTP server: JSON API plus a dependency-free web chat UI."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .generate import load_model_from_checkpoint
from .model import generate_stream
from .tokenizer import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
    EOS_TOKEN,
    SYSTEM_END_TOKEN,
    SYSTEM_TOKEN,
    USER_END_TOKEN,
    USER_TOKEN,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC_FILES = {"app.js": "application/javascript", "style.css": "text/css"}


@dataclass
class GenerateOptions:
    prompt: str = ""
    system: str = DEFAULT_SYSTEM_PROMPT
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.05

    @classmethod
    def from_payload(cls, payload: dict) -> "GenerateOptions":
        def number(name: str, default: float, low: float, high: float) -> float:
            value = payload.get(name, default)
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a number") from exc
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
            return value

        prompt = payload.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return cls(
            prompt=prompt,
            system=str(payload.get("system", DEFAULT_SYSTEM_PROMPT)),
            max_new_tokens=int(number("max_tokens", payload.get("max_new_tokens", 256), 1, 2048)),
            temperature=number("temperature", 0.7, 0.0, 3.0),
            top_k=int(number("top_k", 50, 0, 1000)),
            top_p=number("top_p", 0.95, 0.01, 1.0),
            repetition_penalty=number("repetition_penalty", 1.05, 1.0, 2.0),
        )


def build_chat_prompt(tokenizer, system: str, messages: list[dict]) -> str:
    """Render a conversation with the tokenizer's role markers."""
    parts = [f"{BOS_TOKEN}{SYSTEM_TOKEN}{system}{SYSTEM_END_TOKEN}"]
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"{USER_TOKEN}{content}{USER_END_TOKEN}")
        elif role == "assistant":
            parts.append(f"{ASSISTANT_TOKEN}{content}{EOS_TOKEN}")
    parts.append(ASSISTANT_TOKEN)
    return "".join(parts)


def _checkpoint_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def create_app(
    checkpoint: str,
    tokenizer_path: str | None = None,
    device: str = "auto",
    checkpoint_root: str = "checkpoints",
):
    from fastapi import Body, FastAPI
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

    state: dict[str, Any] = {
        "model": None,
        "tokenizer": None,
        "device": device,
        "checkpoint": str(checkpoint),
        "tokenizer_path": tokenizer_path,
    }
    lock = threading.Lock()

    def load(checkpoint_path: str) -> None:
        model, tokenizer, resolved = load_model_from_checkpoint(
            checkpoint_path, state["tokenizer_path"], state["device"]
        )
        state["model"] = model
        state["tokenizer"] = tokenizer
        state["device"] = str(next(model.parameters()).device)
        state["checkpoint"] = str(checkpoint_path)

    load(checkpoint)

    def list_checkpoints() -> list[str]:
        root = Path(checkpoint_root)
        if not root.exists():
            return []
        found = [
            _checkpoint_name(path, root)
            for pattern in ("*.pt", "*/*.pt")
            for path in sorted(root.glob(pattern))
        ]
        return found

    def prepare_ids(options: GenerateOptions, messages: list[dict] | None) -> torch.Tensor:
        tokenizer = state["tokenizer"]
        if messages:
            prompt = build_chat_prompt(tokenizer, options.system, messages)
        else:
            prompt = tokenizer.format_chat(options.prompt, options.system)
        ids = tokenizer.encode(prompt)
        keep = max(1, state["model"].cfg.max_seq_len - options.max_new_tokens)
        if len(ids) > keep:
            ids = ids[-keep:]
        return torch.tensor([ids], dtype=torch.long, device=state["device"])

    def stream_tokens(options: GenerateOptions, messages: list[dict] | None):
        """Server-sent events: one text delta per token."""
        tokenizer = state["tokenizer"]
        input_ids = prepare_ids(options, messages)
        produced: list[int] = []
        emitted = ""
        with lock:
            for token_id in generate_stream(
                state["model"],
                input_ids,
                max_new_tokens=options.max_new_tokens,
                temperature=options.temperature,
                top_k=options.top_k,
                top_p=options.top_p,
                repetition_penalty=options.repetition_penalty,
                eos_token_id=tokenizer.eos_id,
            ):
                produced.append(token_id)
                # re-decode the whole answer: BPE tokens split words and multi-byte
                # characters, so per-token decoding would produce broken text
                text = tokenizer.decode(produced, skip_special=True)
                if len(text) > len(emitted):
                    delta = text[len(emitted):]
                    emitted = text
                    yield f"data: {json.dumps({'token': delta})}\n\n"
        payload = {
            "done": True,
            "completion_tokens": len(produced),
            "prompt_tokens": int(input_ids.shape[1]),
            "device": state["device"],
        }
        yield f"data: {json.dumps(payload)}\n\n"

    def full_answer(options: GenerateOptions, messages: list[dict] | None) -> dict[str, Any]:
        tokenizer = state["tokenizer"]
        input_ids = prepare_ids(options, messages)
        produced: list[int] = []
        with lock:
            for token_id in generate_stream(
                state["model"],
                input_ids,
                max_new_tokens=options.max_new_tokens,
                temperature=options.temperature,
                top_k=options.top_k,
                top_p=options.top_p,
                repetition_penalty=options.repetition_penalty,
                eos_token_id=tokenizer.eos_id,
            ):
                produced.append(token_id)
        return {
            "text": tokenizer.decode(produced, skip_special=True).strip(),
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": len(produced),
            "device": state["device"],
        }

    app = FastAPI(title="CodePlan-LM", version="0.2.0")

    def health_payload() -> dict[str, Any]:
        model = state["model"]
        return {
            "status": "ok",
            "model": Path(state["checkpoint"]).stem,
            "checkpoint": state["checkpoint"],
            "parameters": model.num_parameters(),
            "vocab_size": state["tokenizer"].vocab_size,
            "device": state["device"],
            "context": model.cfg.max_seq_len,
        }

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/static/{name}")
    def static_file(name: str):
        if name not in STATIC_FILES:
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(WEB_DIR / name, media_type=STATIC_FILES[name])

    @app.get("/health")
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return health_payload()

    @app.get("/api/checkpoints")
    def checkpoints() -> dict[str, Any]:
        return {"checkpoints": list_checkpoints(), "current": state["checkpoint"]}

    @app.post("/api/load")
    def load_endpoint(payload: dict = Body(...)) -> Any:
        name = str(payload.get("checkpoint", "")).strip()
        if not name:
            return JSONResponse({"error": "checkpoint is required"}, status_code=400)
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = Path(checkpoint_root) / name
        if not candidate.exists():
            return JSONResponse({"error": f"checkpoint not found: {candidate}"}, status_code=404)
        try:
            load(str(candidate))
        except Exception as exc:  # torch load errors are not worth a 500 trace
            return JSONResponse({"error": str(exc)}, status_code=400)
        return health_payload()

    @app.post("/api/chat")
    def chat_endpoint(payload: dict = Body(...)) -> Any:
        raw_messages = payload.get("messages")
        messages: list[dict] | None = None
        if raw_messages is not None:
            if not isinstance(raw_messages, list):
                return JSONResponse({"error": "messages must be a list"}, status_code=400)
            messages = [
                {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                for m in raw_messages
                if isinstance(m, dict)
            ]
        try:
            if messages:
                options = GenerateOptions(
                    prompt="".join(m["content"] for m in messages if m["role"] == "user"),
                    system=str(payload.get("system", DEFAULT_SYSTEM_PROMPT)),
                    max_new_tokens=int(payload.get("max_tokens", 256)),
                    temperature=float(payload.get("temperature", 0.7)),
                    top_k=int(payload.get("top_k", 50)),
                    top_p=float(payload.get("top_p", 0.95)),
                )
                if not messages[-1]["content"].strip():
                    return JSONResponse({"error": "last message is empty"}, status_code=400)
            else:
                options = GenerateOptions.from_payload(payload)
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        if payload.get("stream"):
            return StreamingResponse(
                stream_tokens(options, messages),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return full_answer(options, messages)

    @app.post("/generate")
    def generate_endpoint(payload: dict = Body(...)) -> Any:
        try:
            options = GenerateOptions.from_payload(payload)
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return full_answer(options, None)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve CodePlan-LM with a web chat UI.")
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
    displayed = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    print(f"CodePlan-LM GUI: http://{displayed}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
