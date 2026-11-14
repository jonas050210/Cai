import json
import shutil

import pytest

from codeplan.agent import Toolbox, parse_tool_call, run_session
from codeplan.sandbox import Sandbox, SandboxUnavailable

HAVE_BWRAP = bool(shutil.which("bwrap") and shutil.which("prlimit"))


def test_sandbox_refuses_without_bubblewrap(tmp_path, monkeypatch):
    monkeypatch.setattr(Sandbox, "backend", staticmethod(lambda: None))
    box = Sandbox(tmp_path)
    with pytest.raises(SandboxUnavailable):
        box.command("echo hi")


def test_sandbox_command_contains_isolation_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(Sandbox, "backend", staticmethod(lambda: "bubblewrap"))
    box = Sandbox(tmp_path, allow_network=False)
    argv = box.command("echo hi")
    joined = " ".join(argv)
    assert "bwrap" in joined
    assert "--unshare-net" in joined
    assert f"--bind {tmp_path.resolve()} /workspace" in joined
    assert "--die-with-parent" in joined


def test_network_flag_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(Sandbox, "backend", staticmethod(lambda: "bubblewrap"))
    assert "--unshare-net" not in " ".join(Sandbox(tmp_path, allow_network=True).command("x"))


@pytest.mark.skipif(not HAVE_BWRAP, reason="bubblewrap not installed")
def test_sandbox_runs_and_cannot_see_outside(tmp_path):
    box = Sandbox(tmp_path)
    ok = box.run("echo hello && ls /workspace")
    assert ok.ok and "hello" in ok.stdout
    outside = box.run("ls /home")
    assert not outside.ok or "No such file" in outside.stderr or outside.stdout.strip() in ("", "..", ".")


def test_parse_tool_call_finds_first_json_object():
    assert parse_tool_call('text {"tool": "run", "args": {"command": "ls"}} more') == {
        "tool": "run", "args": {"command": "ls"}
    }
    assert parse_tool_call('{"tool": "done", "args": {}}')["tool"] == "done"
    assert parse_tool_call("no json here") is None


def test_toolbox_write_read_list_stay_in_workspace(tmp_path):
    box = Toolbox(tmp_path)
    assert box.write("sub/a.txt", "hello")[0]
    assert box.read("sub/a.txt") == (True, "hello")
    assert "sub/" in box.list(".")[1]
    ok, message = box.read("../../etc/passwd")
    assert not ok
    assert "escapes the workspace" in message


def test_session_runs_scripted_policy(tmp_path):
    written = []

    def policy(transcript, steps):
        if not steps:
            return json.dumps({"tool": "write", "args": {"path": "notes.md", "text": "# hi"}})
        if len(steps) == 1:
            return json.dumps({"tool": "read", "args": {"path": "notes.md"}})
        return json.dumps({"tool": "done", "args": {"summary": "wrote and read a file"}})

    session = run_session("write a note", tmp_path, policy, Toolbox(tmp_path))
    assert session.finished
    assert [step.tool for step in session.steps] == ["write", "read", "done"]
    assert session.summary == "wrote and read a file"
    assert (tmp_path / "notes.md").read_text() == "# hi"
    written.append(True)


def test_session_stops_at_step_budget(tmp_path):
    def policy(transcript, steps):
        return json.dumps({"tool": "list", "args": {"path": "."}})

    session = run_session("loop", tmp_path, policy, Toolbox(tmp_path), max_steps=3)
    assert not session.finished
    assert len(session.steps) == 3


def test_invalid_reply_is_recorded(tmp_path):
    calls = {"n": 0}

    def policy(transcript, steps):
        calls["n"] += 1
        return "I will just think about it." if calls["n"] == 1 else json.dumps(
            {"tool": "done", "args": {"summary": "ok"}}
        )

    session = run_session("x", tmp_path, policy, Toolbox(tmp_path))
    assert session.steps[0].tool == "error"
    assert session.finished


def test_prompt_is_rendered_from_the_tool_table():
    from codeplan.agent import SYSTEM_PROMPT, TOOLS

    assert len({spec.name for spec in TOOLS}) == len(TOOLS)
    for spec in TOOLS:
        assert spec.name in SYSTEM_PROMPT
        assert spec.args in SYSTEM_PROMPT


def test_edit_requires_a_unique_match(tmp_path):
    from codeplan.agent import Toolbox

    (tmp_path / "a.py").write_text("x = 1\ny = 1\n", encoding="utf-8")
    box = Toolbox(tmp_path)
    ok, message = box.edit("a.py", "1", "2")
    assert not ok and "2 times" in message
    ok, message = box.edit("a.py", "x = 1", "x = 42")
    assert ok
    assert (tmp_path / "a.py").read_text().startswith("x = 42")
    ok, message = box.edit("a.py", "zzz", "1")
    assert not ok and "not found" in message


def test_grep_and_glob_stay_within_workspace(tmp_path):
    from codeplan.agent import Toolbox

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 'needle'\n", encoding="utf-8")
    box = Toolbox(tmp_path)
    ok, hits = box.grep("needle")
    assert ok and "src/app.py:2" in hits
    ok, files = box.glob("**/*.py")
    assert ok and files.strip() == "src/app.py"
    ok, message = box.grep("x", "../")
    assert not ok and "escapes" in message


def test_unknown_tool_lists_available_tools(tmp_path):
    from codeplan.agent import Toolbox

    ok, message = Toolbox(tmp_path).call("sudo", {})
    assert not ok
    assert "available:" in message
    assert "grep" in message


def test_observations_are_clipped(tmp_path):
    from codeplan.agent import Toolbox

    (tmp_path / "big.txt").write_text("a" * 20_000, encoding="utf-8")
    ok, text = Toolbox(tmp_path, max_observation=500).read("big.txt")
    assert ok
    assert len(text) < 700
    assert "truncated" in text


def test_transcript_compresses_old_steps(tmp_path):
    from codeplan.agent import Step, build_transcript

    steps = [
        Step("list", {"path": "."}, "a\nb\nc", True),
        Step("read", {"path": "a"}, "content " * 100, True),
        Step("write", {"path": "b"}, "wrote 3 characters", True),
        Step("grep", {"pattern": "x"}, "(no matches)", True),
        Step("run", {"command": "ls"}, "a b", True),
    ]
    transcript = build_transcript("do something", steps, keep=2)
    assert "Earlier steps (compressed)" in transcript
    assert "content " * 100 not in transcript  # old observation collapsed
    assert "a b" in transcript  # recent one is verbatim


def test_session_log_roundtrip(tmp_path):
    import json

    from codeplan.agent import run_session

    def policy(transcript, steps):
        return json.dumps({"tool": "done", "args": {"summary": "nothing to do"}})

    session = run_session("task", tmp_path, policy)
    path = session.save()
    payload = json.loads(path.read_text())
    assert payload["finished"] is True
    assert payload["steps"][0]["tool"] == "done"
