import json
import threading
import time

from codeplan import collect as collect_mod
from codeplan.collect import collect


def _repo(name, size_kb=1000):
    return {
        "full_name": name,
        "license": "mit",
        "stars": 5000,
        "size_kb": size_kb,
        "branch": "main",
        "language": "Python",
    }


def _record(repo, path):
    return {
        "id": f"{repo}:{path}",
        "text": f"def {repo.replace('/', '_')}_{path.replace('.', '_')}():\n    return 1\n" * 8,
        "lang": "en",
        "domain": "code",
        "license": "mit",
        "source": f"github:{repo}/{path}",
    }


def test_parallel_and_sequential_write_the_same_records(tmp_path, monkeypatch):
    names = [f"owner{i}/repo{i}" for i in range(6)]
    monkeypatch.setattr(
        collect_mod, "fetch_candidates",
        lambda repos, token, sleep, workers: ([_repo(n) for n in names], [], 0),
    )
    seen_threads = set()

    def fake_download(repo, token, max_repo_mb):
        seen_threads.add(threading.current_thread().name)
        time.sleep(0.01)  # give the pool a chance to spread across workers
        records = [_record(repo["full_name"], "a.py"), _record(repo["full_name"], "b.py")]
        return records, "digest", 2

    monkeypatch.setattr(collect_mod, "download_records", fake_download)

    seq = collect(
        tmp_path / "seq.jsonl", ["python"], ["mit"], 0, 100, 40, None, 0.0,
        repos=names, min_records_per_repo=1, workers=1,
    )
    threads_before = set(seen_threads)
    seen_threads.clear()

    par = collect(
        tmp_path / "par.jsonl", ["python"], ["mit"], 0, 100, 40, None, 0.0,
        repos=names, min_records_per_repo=1, workers=4,
    )

    assert seq["records"] == par["records"] == 12
    lines_seq = (tmp_path / "seq.jsonl").read_text().splitlines()
    lines_par = (tmp_path / "par.jsonl").read_text().splitlines()
    # same records; only the line order differs (parallel = completion order,
    # which does not matter because prepare shuffles and splits afterwards)
    sources_seq = sorted(json.loads(line)["source"] for line in lines_seq)
    sources_par = sorted(json.loads(line)["source"] for line in lines_par)
    assert sources_seq == sources_par
    # the sequential run downloads on the main thread, the parallel one on pool
    # threads (asserting "more than one thread" would be flaky for tiny tasks)
    assert threads_before == {"MainThread"}
    assert "MainThread" not in seen_threads


def test_parallel_run_dedupes_identical_text_across_repos(tmp_path, monkeypatch):
    names = ["a/one", "b/two", "c/three"]
    monkeypatch.setattr(
        collect_mod, "fetch_candidates",
        lambda repos, token, sleep, workers: ([_repo(n) for n in names], [], 0),
    )
    same = _record("shared", "same.py")
    monkeypatch.setattr(
        collect_mod, "download_records",
        lambda repo, token, max_repo_mb: ([dict(same)], "digest", 1),
    )
    summary = collect(
        tmp_path / "out.jsonl", ["python"], ["mit"], 0, 100, 40, None, 0.0,
        repos=names, min_records_per_repo=1, workers=3,
    )
    # first repo stores it, the other two are deduped down to zero records
    assert summary["records"] == 1


def test_parallel_path_skips_small_and_huge_repos(tmp_path, monkeypatch):
    repos = [_repo("ok/one"), _repo("tiny/two", size_kb=5), _repo("huge/three", size_kb=10**9)]
    monkeypatch.setattr(
        collect_mod, "fetch_candidates",
        lambda names, token, sleep, workers: (repos, [], 0),
    )
    monkeypatch.setattr(
        collect_mod, "download_records",
        lambda repo, token, max_repo_mb: ([_record(repo["full_name"], "a.py")], "d", 1),
    )
    summary = collect(
        tmp_path / "out.jsonl", ["python"], ["mit"], 0, 100, 40, None, 0.0,
        repos=[r["full_name"] for r in repos], min_records_per_repo=1, workers=4,
    )
    assert summary["records"] == 1
    assert summary["skipped"] == 2
    statuses = [
        json.loads(line)["status"]
        for line in (tmp_path / "out.manifest.jsonl").read_text().splitlines()
    ]
    assert "skipped-small" in statuses and "skipped-huge" in statuses


def test_resume_skips_repos_already_in_the_manifest(tmp_path, monkeypatch):
    names = ["a/one", "b/two", "c/three"]
    monkeypatch.setattr(
        collect_mod, "fetch_candidates",
        lambda repos, token, sleep, workers: ([_repo(n) for n in names], [], 0),
    )
    calls = []

    def fake_download(repo, token, max_repo_mb):
        calls.append(repo["full_name"])
        return [_record(repo["full_name"], "a.py")], "digest", 1

    monkeypatch.setattr(collect_mod, "download_records", fake_download)

    out = tmp_path / "out.jsonl"
    first = collect(
        out, ["python"], ["mit"], 0, 100, 40, None, 0.0,
        repos=names, min_records_per_repo=1, workers=2,
    )
    assert first["records"] == 3
    assert sorted(calls) == names

    # a second run must not download anything again
    calls.clear()
    second = collect(
        out, ["python"], ["mit"], 0, 100, 40, None, 0.0,
        repos=names, min_records_per_repo=1, workers=2,
    )
    assert calls == []
    assert second["already_done"] == 3
    assert second["records"] == 0
    # and the existing records are still there (append, not truncate)
    assert len(out.read_text().splitlines()) == 3

    # --fresh starts over
    calls.clear()
    third = collect(
        out, ["python"], ["mit"], 0, 100, 40, None, 0.0,
        repos=names, min_records_per_repo=1, workers=2, fresh=True,
    )
    assert sorted(calls) == names
    assert third["records"] == 3
    assert len(out.read_text().splitlines()) == 3


def test_one_failing_repo_does_not_end_the_run(tmp_path, monkeypatch):
    import http.client

    names = ["ok/one", "broken/two", "ok/three"]
    monkeypatch.setattr(
        collect_mod, "fetch_candidates",
        lambda repos, token, sleep, workers: ([_repo(n) for n in names], [], 0),
    )

    def flaky(repo, token, max_repo_mb):
        if repo["full_name"] == "broken/two":
            raise http.client.IncompleteRead(b"", 27294)
        return [_record(repo["full_name"], "a.py")], "digest", 1

    monkeypatch.setattr(collect_mod, "download_records", flaky)
    summary = collect(
        tmp_path / "out.jsonl", ["python"], ["mit"], 0, 100, 40, None, 0.0,
        repos=names, min_records_per_repo=1, workers=3,
    )
    assert summary["records"] == 2
    assert summary["errors"] == 1
    errors = [
        json.loads(line)
        for line in (tmp_path / "out.manifest.jsonl").read_text().splitlines()
        if json.loads(line)["status"] == "error"
    ]
    assert errors[0]["full_name"] == "broken/two"
    assert "IncompleteRead" in errors[0]["error"]


def test_download_retries_truncated_responses(monkeypatch):
    import http.client

    from codeplan.collect import _download

    attempts = {"n": 0}

    class FakeResponse:
        def __init__(self):
            self.chunks = [b"x" * 10, b""]

        def read(self, size):
            if attempts["n"] == 1:
                raise http.client.IncompleteRead(b"", 5)
            return self.chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=0):
        attempts["n"] += 1
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)

    assert _download("http://example.invalid/x", None, 1024) == b"x" * 10
    assert attempts["n"] == 2  # first attempt failed, second succeeded


def test_download_gives_up_after_three_attempts(monkeypatch):
    import http.client

    from codeplan.collect import _download

    def always_fail(request, timeout=0):
        raise http.client.IncompleteRead(b"", 5)

    monkeypatch.setattr("urllib.request.urlopen", always_fail)
    monkeypatch.setattr("time.sleep", lambda s: None)
    try:
        _download("http://example.invalid/x", None, 1024)
    except http.client.IncompleteRead:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected the transient error to surface")
