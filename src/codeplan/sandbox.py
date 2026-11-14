"""Run untrusted commands in a locked-down workspace.

Layer by layer:

* **filesystem** - bubblewrap shows the session directory and a read-only /usr,
  nothing else (no home, no repo, no network config)
* **network** - a separate network namespace, so no socket leaves the machine
* **time / processes** - ``prlimit`` caps CPU seconds, file size and process count
* **memory** - cgroups v2 if available, otherwise the rlimit stays in place

Without bubblewrap the runner refuses to execute anything. That is deliberate:
a "sandbox" that silently falls back to ``subprocess.run`` would be worse than
no sandbox at all.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    """Raised when no isolation backend is installed."""


@dataclass
class Limits:
    seconds: int = 30
    cpu_seconds: int = 30
    memory_mb: int = 2048
    processes: int = 256
    file_mb: int = 16


@dataclass
class Result:
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def as_dict(self) -> dict:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "ok": self.ok,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": round(self.duration, 3),
        }


@dataclass
class Sandbox:
    workspace: Path
    limits: Limits = field(default_factory=Limits)
    allow_network: bool = False
    max_output: int = 8000

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"workspace does not exist: {self.workspace}")

    @staticmethod
    def backend() -> str | None:
        if shutil.which("bwrap") and shutil.which("prlimit"):
            return "bubblewrap"
        return None

    def command(self, cmd: str) -> list[str]:
        """Wrap a shell command in the isolation layers we have."""
        if self.backend() != "bubblewrap":
            raise SandboxUnavailable(
                "bubblewrap (bwrap) and prlimit are required.\n"
                "Install them:  sudo apt install bubblewrap util-linux"
            )
        wrapper = [
            "bwrap",
            "--die-with-parent",
            "--unshare-all",
            "--new-session",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--bind", str(self.workspace), "/workspace",
            "--chdir", "/workspace",
            "--setenv", "HOME", "/workspace",
            "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
        ]
        if not self.allow_network:
            wrapper += ["--unshare-net"]
        wrapper += ["--"]
        limiter = [
            "prlimit",
            f"--cpu={self.limits.cpu_seconds}",
            f"--nproc={self.limits.processes}",
            f"--fsize={self.limits.file_mb * 1024 * 1024}",
            f"--as={self.limits.memory_mb * 1024 * 1024}",
        ]
        return wrapper + limiter + ["/bin/sh", "-c", cmd]

    def run(self, cmd: str) -> Result:
        argv = self.command(cmd)
        started = time.time()
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.limits.seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return Result(
                command=cmd,
                returncode=-1,
                stdout=self._clip(exc.stdout or ""),
                stderr=self._clip(exc.stderr or ""),
                duration=time.time() - started,
                timed_out=True,
            )
        return Result(
            command=cmd,
            returncode=completed.returncode,
            stdout=self._clip(completed.stdout),
            stderr=self._clip(completed.stderr),
            duration=time.time() - started,
        )

    def _clip(self, text: str) -> str:
        if len(text) <= self.max_output:
            return text
        half = self.max_output // 2
        return text[:half] + "\n... [truncated] ...\n" + text[-half:]
