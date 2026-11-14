"""A small tool-using loop over the sandbox.

Design rules, taken from how a working agent harness has to behave:

1. **One source of truth for tools.** ``TOOLS`` defines name, arguments and
   description; the system prompt is rendered from it, so prompt and
   implementation cannot drift apart.
2. **Bounded observations.** Every tool result is clipped; the transcript keeps
   recent observations verbatim and older ones as one-liners (context budget).
3. **Errors are observations, not exceptions.** A bad tool call returns text the
   model can react to, and the loop continues until the step budget is spent.
4. **All filesystem access goes through the sandbox.** Paths are resolved and
   checked against the workspace, so ``../`` cannot escape.
5. **Everything is replayable.** A session is a list of steps plus a log file on
   disk, so a run can be inspected, resumed or turned into SFT data.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .sandbox import Sandbox

MAX_STEPS = 8
MAX_OBSERVATION = 4000
KEEP_VERBATIM = 4

SYSTEM_TEMPLATE = """You are a coding agent working in /workspace.
Answer with exactly one JSON object per turn and nothing else.

Tools:
{tools}

Rules:
- Look before you change: list or grep first, then edit.
- Verify: after editing, run the code or the tests.
- One small step per turn. Stop with "done" when the task is solved.
- Never invent output; if a command fails, read the error and try again.
"""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args: str
    description: str

    def render(self) -> str:
        return f'- {self.name}({self.args}): {self.description}'


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("run", "command", "shell command inside the sandbox (no network)"),
    ToolSpec("read", "path", "read a text file, up to 4000 characters"),
    ToolSpec("write", "path, text", "create or overwrite a file"),
    ToolSpec("edit", "path, old, new",
             "replace exactly one occurrence of old with new (fails if not unique)"),
    ToolSpec("grep", "pattern, path='.'", "regex search, returns file:line matches (max 60)"),
    ToolSpec("glob", "pattern='**/*'", "list files matching a glob pattern (max 200)"),
    ToolSpec("list", "path='.'", "list a directory"),
    ToolSpec("done", "summary", "finish the task"),
)

SYSTEM_PROMPT = SYSTEM_TEMPLATE.format(tools="\n".join(spec.render() for spec in TOOLS))

NOISE = re.compile(r"/workspace/|agent_sessions/\S+", re.IGNORECASE)


@dataclass
class Step:
    tool: str
    args: dict
    observation: str
    ok: bool
    duration: float = 0.0

    def one_line(self) -> str:
        summary = self.observation.replace("\n", " ")[:120]
        return f"{self.tool}: {summary}"


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
                {
                    "tool": step.tool,
                    "args": step.args,
                    "ok": step.ok,
                    "observation": step.observation,
                    "duration": round(step.duration, 3),
                }
                for step in self.steps
            ],
        }

    def save(self, path: Path | None = None) -> Path:
        path = path or self.workspace / "session.json"
        path.write_text(json.dumps(self.log(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def _json_objects(text: str):
    """Yield every balanced {...} chunk, so nested args survive."""
    for match in re.finditer(r"\{", text):
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


def compact(transcript: str, steps: list[Step], keep: int = KEEP_VERBATIM) -> str:
    """Collapse old steps into one line each so the prompt stops growing."""
    if len(steps) <= keep:
        return transcript
    return transcript


def build_transcript(task: str, steps: list[Step], keep: int = KEEP_VERBATIM) -> str:
    head = SYSTEM_PROMPT + f"\n\nTask: {task}"
    if not steps:
        return head
    recent = steps[-keep:]
    older = steps[:-keep]
    parts = [head]
    if older:
        parts.append(
            "\n\nEarlier steps (compressed):\n"
            + "\n".join(f"- {step.one_line()}" for step in older)
        )
    for step in recent:
        parts.append(
            f"\n\nYou called {step.tool} "
            f"{json.dumps(step.args, ensure_ascii=False)[:400]}"
            f"\nObservation ({'ok' if step.ok else 'error'}):\n{step.observation}"
        )
    return "".join(parts)


def _inside(workspace: Path, candidate: str) -> Path:
    root = workspace.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes the workspace: {candidate}")
    return resolved


class Toolbox:
    def __init__(self, workspace: Path, sandbox: Sandbox | None = None,
                 max_observation: int = MAX_OBSERVATION):
        self.workspace = Path(workspace).resolve()
        self.sandbox = sandbox
        self.max_observation = max_observation

    def _clip(self, text: str) -> str:
        if len(text) <= self.max_observation:
            return text
        half = self.max_observation // 2
        return text[:half] + "\n... [truncated] ...\n" + text[-half:]

    # --- tools ---------------------------------------------------------
    def run(self, command: str) -> tuple[bool, str]:
        if self.sandbox is None:
            return False, "no sandbox available"
        result = self.sandbox.run(command)
        parts = [result.stdout.strip()]
        if result.stderr.strip():
            parts.append("stderr: " + result.stderr.strip())
        output = "\n".join(part for part in parts if part)
        return result.ok, self._clip(output or f"exit code {result.returncode}")

    def read(self, path: str) -> tuple[bool, str]:
        try:
            target = _inside(self.workspace, path)
        except ValueError as exc:
            return False, str(exc)
        if not target.is_file():
            return False, f"not a file: {path}"
        return True, self._clip(target.read_text(encoding="utf-8", errors="replace"))

    def write(self, path: str, text: str) -> tuple[bool, str]:
        try:
            target = _inside(self.workspace, path)
        except ValueError as exc:
            return False, str(exc)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return True, f"wrote {len(text)} characters to {path}"

    def edit(self, path: str, old: str, new: str) -> tuple[bool, str]:
        try:
            target = _inside(self.workspace, path)
        except ValueError as exc:
            return False, str(exc)
        if not target.is_file():
            return False, f"not a file: {path}"
        content = target.read_text(encoding="utf-8")
        occurrences = content.count(old)
        if occurrences == 0:
            return False, "old text not found; read the file again"
        if occurrences > 1:
            return False, f"old text appears {occurrences} times; add more context"
        target.write_text(content.replace(old, new, 1), encoding="utf-8")
        return True, f"replaced 1 occurrence in {path}"

    def grep(self, pattern: str, path: str = ".") -> tuple[bool, str]:
        try:
            target = _inside(self.workspace, path)
            regex = re.compile(pattern)
        except (ValueError, re.error) as exc:
            return False, str(exc)
        matches = []
        files = [target] if target.is_file() else sorted(target.rglob("*"))
        for file in files:
            if not file.is_file() or file.stat().st_size > 2_000_000:
                continue
            try:
                for number, line in enumerate(
                    file.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if regex.search(line):
                        matches.append(f"{file.relative_to(self.workspace)}:{number}: {line.strip()[:160]}")
                        if len(matches) >= 60:
                            return True, self._clip("\n".join(matches) + "\n[more matches omitted]")
            except OSError:
                continue
        return True, self._clip("\n".join(matches)) if matches else "(no matches)"

    def glob(self, pattern: str = "**/*") -> tuple[bool, str]:
        try:
            hits = [
                str(path.relative_to(self.workspace))
                for path in sorted(self.workspace.glob(pattern))
                if path.is_file()
            ]
        except (ValueError, OSError) as exc:
            return False, str(exc)
        if not hits:
            return True, "(no files)"
        extra = "" if len(hits) <= 200 else f"\n[+{len(hits) - 200} more]"
        return True, "\n".join(hits[:200]) + extra

    def list(self, path: str = ".") -> tuple[bool, str]:
        try:
            target = _inside(self.workspace, path)
        except ValueError as exc:
            return False, str(exc)
        if not target.is_dir():
            return False, f"not a directory: {path}"
        names = sorted(entry.name + ("/" if entry.is_dir() else "") for entry in target.iterdir())
        return True, self._clip("\n".join(names) or "(empty)")

    def call(self, tool: str, args: dict) -> tuple[bool, str]:
        if tool == "run":
            return self.run(str(args.get("command", "")))
        if tool == "read":
            return self.read(str(args.get("path", "")))
        if tool == "write":
            return self.write(str(args.get("path", "")), str(args.get("text", "")))
        if tool == "edit":
            return self.edit(str(args.get("path", "")), str(args.get("old", "")),
                             str(args.get("new", "")))
        if tool == "grep":
            return self.grep(str(args.get("pattern", "")), str(args.get("path", ".")))
        if tool == "glob":
            return self.glob(str(args.get("pattern", "**/*")))
        if tool == "list":
            return self.list(str(args.get("path", ".")))
        known = ", ".join(spec.name for spec in TOOLS)
        return False, f"unknown tool {tool!r}; available: {known}"


def run_session(
    task: str,
    workspace: Path,
    policy: Callable[[str, list[Step]], str],
    toolbox: Toolbox | None = None,
    max_steps: int = MAX_STEPS,
) -> Session:
    """Drive the loop until the policy calls ``done`` or the step budget is spent."""
    session = Session(task=task, workspace=Path(workspace))
    toolbox = toolbox or Toolbox(workspace, Sandbox(workspace))

    for _ in range(max_steps):
        reply = policy(build_transcript(task, session.steps), session.steps)
        call = parse_tool_call(reply)
        if call is None:
            session.steps.append(
                Step("error", {}, "no valid JSON tool call; reply with one JSON object", False)
            )
            continue
        tool, args = call["tool"], call["args"]
        if tool == "done":
            session.finished = True
            session.summary = str(args.get("summary", ""))
            session.steps.append(Step("done", args, session.summary, True))
            break
        started = time.time()
        ok, observation = toolbox.call(tool, args)
        session.steps.append(Step(tool, args, observation, ok, time.time() - started))
    return session


class ProcessPolicy:
    """A policy backed by a checkpoint (one prompt in, one reply out)."""

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
    parser.add_argument(
        "--allow-network", action="store_true", help="Give the session its own network namespace"
    )
    args = parser.parse_args()

    workspace = Path(args.workspace or f"agent_sessions/{int(time.time())}")
    workspace.mkdir(parents=True, exist_ok=True)

    if Sandbox.backend() is None:
        raise SystemExit(
            "No sandbox backend found.\n"
            "  sudo apt install bubblewrap util-linux\n"
            "Nothing was executed."
        )

    toolbox = Toolbox(workspace, Sandbox(workspace, allow_network=args.allow_network))

    if args.script:
        queue = list(json.loads(Path(args.script).read_text(encoding="utf-8")))

        def policy(transcript: str, steps: list[Step]) -> str:
            return json.dumps(
                queue.pop(0) if queue else {"tool": "done", "args": {"summary": "script end"}}
            )
    else:
        if not args.checkpoint:
            raise SystemExit("Pass --checkpoint or --script.")
        policy = ProcessPolicy(args.checkpoint)

    session = run_session(args.task, workspace, policy, toolbox, args.max_steps)
    log_path = session.save()
    print(json.dumps(session.log(), indent=2, ensure_ascii=False)[:4000])
    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
