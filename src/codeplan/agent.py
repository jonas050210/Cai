"""A small tool-using loop over the sandbox.

The model (or a scripted policy, which is how this is tested) answers with one
JSON object per turn:

    {"tool": "run", "args": {"command": "pytest -q"}}

Tools are deliberately few and boring. Everything that touches the filesystem
goes through the sandbox, so a session cannot reach outside its workspace.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .sandbox import Sandbox

MAX_STEPS = 8
TOOL_CALL_RE = re.compile(r"\{")

SYSTEM_PROMPT = (
    "You are a coding agent working in /workspace. Answer with exactly one JSON "
    'object per turn: {"tool": "<name>", "args": {...}}. Available tools:\n'
    '- run {"command": "..."} - shell command in the sandbox\n'
    '- read {"path": "..."} - read a file\n'
    '- write {"path": "...", "text": "..."} - write a file\n'
    '- list {"path": "."} - list a directory\n'
    '- done {"summary": "..."} - finish the task\n'
    "Use small steps and check the result before the next call."
)


@dataclass
class Step:
    tool: str
    args: dict
    observation: str
    ok: bool


@dataclass
class Session:
    task: str
    workspace: Path
    steps: list[Step] = field(default_factory=list)
    finished: bool = False
    summary: str = ""

    def log(self) -> dict:
        return {
            "task": self.task,
            "workspace": str(self.workspace),
            "finished": self.finished,
            "summary": self.summary,
            "steps": [
                {"tool": s.tool, "args": s.args, "ok": s.ok, "observation": s.observation}
                for s in self.steps
            ],
        }


def _json_objects(text: str):
    """Yield every balanced {...} chunk, so nested args survive."""
    for match in TOOL_CALL_RE.finditer(text):
        start = match.start()
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def parse_tool_call(text: str) -> dict | None:
    """Pull the first complete JSON tool call out of a model reply."""
    for chunk in _json_objects(text):
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("tool"), str):
            args = payload.get("args")
            return {"tool": payload["tool"], "args": args if isinstance(args, dict) else {}}
    return None


def _inside(workspace: Path, candidate: str) -> Path:
    resolved = (workspace / candidate).resolve()
    if workspace.resolve() not in resolved.parents and resolved != workspace.resolve():
        raise ValueError("path escapes the workspace")
    return resolved


class Toolbox:
    def __init__(self, workspace: Path, sandbox: Sandbox | None = None):
        self.workspace = Path(workspace).resolve()
        self.sandbox = sandbox

    def run(self, command: str) -> tuple[bool, str]:
        if self.sandbox is None:
            return False, "no sandbox available"
        result = self.sandbox.run(command)
        parts = [result.stdout.strip()]
        if result.stderr.strip():
            parts.append("stderr: " + result.stderr.strip())
        output = "\n".join(part for part in parts if part)
        return result.ok, output or f"exit code {result.returncode}"

    def read(self, path: str) -> tuple[bool, str]:
        try:
            target = _inside(self.workspace, path)
        except ValueError as exc:
            return False, str(exc)
        if not target.is_file():
            return False, f"not a file: {path}"
        return True, target.read_text(encoding="utf-8", errors="replace")[:8000]

    def write(self, path: str, text: str) -> tuple[bool, str]:
        try:
            target = _inside(self.workspace, path)
        except ValueError as exc:
            return False, str(exc)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return True, f"wrote {len(text)} characters to {path}"

    def list(self, path: str = ".") -> tuple[bool, str]:
        try:
            target = _inside(self.workspace, path)
        except ValueError as exc:
            return False, str(exc)
        if not target.is_dir():
            return False, f"not a directory: {path}"
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return True, "\n".join(names) or "(empty)"

    def call(self, tool: str, args: dict) -> tuple[bool, str]:
        if tool == "run":
            return self.run(str(args.get("command", "")))
        if tool == "read":
            return self.read(str(args.get("path", "")))
        if tool == "write":
            return self.write(str(args.get("path", "")), str(args.get("text", "")))
        if tool == "list":
            return self.list(str(args.get("path", ".")))
        return False, f"unknown tool: {tool}"


def run_session(
    task: str,
    workspace: Path,
    policy: Callable[[str, list[Step]], str],
    toolbox: Toolbox | None = None,
    max_steps: int = MAX_STEPS,
) -> Session:
    """Drive the loop until the policy calls ``done`` or the budget runs out."""
    session = Session(task=task, workspace=Path(workspace))
    toolbox = toolbox or Toolbox(workspace, Sandbox(workspace))
    transcript = SYSTEM_PROMPT + f"\n\nTask: {task}"

    for _ in range(max_steps):
        reply = policy(transcript, session.steps)
        call = parse_tool_call(reply)
        if call is None:
            session.steps.append(Step("error", {}, "no valid JSON tool call", False))
            transcript += "\n\nYour reply had no tool call. Reply with JSON only."
            continue
        tool, args = call["tool"], call["args"]
        if tool == "done":
            session.finished = True
            session.summary = str(args.get("summary", ""))
            session.steps.append(Step("done", args, session.summary, True))
            break
        ok, observation = toolbox.call(tool, args)
        session.steps.append(Step(tool, args, observation, ok))
        transcript += (
            f"\n\nYou called {tool} {json.dumps(args, ensure_ascii=False)}"
            f"\nObservation ({'ok' if ok else 'error'}):\n{observation}"
        )
    return session


class ProcessPolicy:
    """A policy backed by ``codeplan-generate`` (one prompt, one reply)."""

    def __init__(self, checkpoint: str, tokenizer: str | None = None, max_new_tokens: int = 256):
        from .generate import load_model_from_checkpoint
        from .model import generate

        self.model, self.tokenizer, _ = load_model_from_checkpoint(checkpoint, tokenizer)
        self.generate = generate
        self.max_new_tokens = max_new_tokens

    def __call__(self, transcript: str, steps: list[Step]) -> str:
        ids = self.tokenizer.encode(transcript, add_bos=True)
        out = self.generate(
            self.model, ids, max_new_tokens=self.max_new_tokens, temperature=0.2, top_k=40
        )
        return self.tokenizer.decode(out[len(ids):])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tool-using session in a sandbox.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--workspace", default=None, help="Session directory (created if missing)")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint; omit for a demo")
    parser.add_argument("--script", default=None, help="JSON list of tool calls to replay")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    workspace = Path(args.workspace or f"agent_sessions/{int(__import__('time').time())}")
    workspace.mkdir(parents=True, exist_ok=True)

    if Sandbox.backend() is None:
        raise SystemExit(
            "No sandbox backend found.\n"
            "  sudo apt install bubblewrap util-linux\n"
            "Nothing was executed."
        )

    toolbox = Toolbox(workspace, Sandbox(workspace))

    if args.script:
        calls = json.loads(Path(args.script).read_text(encoding="utf-8"))
        queue = list(calls)

        def policy(transcript: str, steps: list[Step]) -> str:
            return json.dumps(queue.pop(0)) if queue else json.dumps({"tool": "done", "args": {}})
    else:
        if not args.checkpoint:
            raise SystemExit("Pass --checkpoint or --script.")
        policy = ProcessPolicy(args.checkpoint)

    session = run_session(args.task, workspace, policy, toolbox, args.max_steps)
    log_path = workspace / "session.json"
    log_path.write_text(json.dumps(session.log(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(session.log(), indent=2, ensure_ascii=False)[:4000])
    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
