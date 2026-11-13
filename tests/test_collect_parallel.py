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
