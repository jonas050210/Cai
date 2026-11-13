"""One-shot generation from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .model import CodePlanLM, generate
from .tokenizer import DEFAULT_SYSTEM_PROMPT, CodePlanTokenizer
from .utils import get_device


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path | None = None,
    device: str = "auto",
) -> tuple[CodePlanLM, CodePlanTokenizer, dict]:
    device_t = get_device(device)
    payload = torch.load(checkpoint_path, map_location=device_t, weights_only=False)
    config = payload.get("config", {})
    model_cfg = ModelConfig.from_dict(config.get("model", {}))

    if tokenizer_path is None:
        candidates = [
            Path(checkpoint_path).parent / "tokenizer.json",
            Path(config.get("tokenizer_path", "artifacts/tokenizer.json")),
        ]
        for cand in candidates:
            if cand.exists():
                tokenizer_path = cand
                break
    if tokenizer_path is None:
        raise FileNotFoundError("Tokenizer not found. Pass --tokenizer explicitly.")

    tokenizer = CodePlanTokenizer.load(tokenizer_path)
    model_cfg.vocab_size = tokenizer.vocab_size
    model = CodePlanLM(model_cfg).to(device_t)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, tokenizer, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from CodePlan-LM.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw", action="store_true", help="Do not wrap prompt in chat template")
    args = parser.parse_args()

    model, tokenizer, _ = load_model_from_checkpoint(args.checkpoint, args.tokenizer, args.device)
    device = next(model.parameters()).device

    prompt = args.prompt if args.raw else tokenizer.format_chat(args.prompt, args.system)
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    out = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_id,
    )
    new_tokens = out[0, input_ids.shape[1]:].tolist()
    print(tokenizer.decode(new_tokens, skip_special=True))


if __name__ == "__main__":
    main()
