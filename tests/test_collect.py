from codeplan.collect import (
    iter_repo_files,
    license_ok,
    records_from_files,
    should_skip_path,
)

PY_CODE = "def add(a, b):\n    return a + b\n" * 20
README = "# Project\n\nThis is a small tool for planning software work.\n" * 6


def test_license_whitelist():
    assert license_ok("MIT")
    assert license_ok("apache-2.0")
    assert not license_ok("GPL-3.0")
    assert not license_ok("NOASSERTION")
    assert not license_ok(None)


def test_should_skip_path():
    assert should_skip_path("node_modules/pkg/index.js")
    assert should_skip_path(".git/config")
    assert should_skip_path("package-lock.json")
    assert should_skip_path("test_helpers.py")
    assert should_skip_path("assets/logo.svg")
    assert not should_skip_path("src/app.py")
    assert not should_skip_path("docs/PLAN.md")


def test_iter_repo_files_filters_binaries_and_big_files():
    members = [
        ("repo/src/app.py", PY_CODE.encode()),
        ("repo/README.md", README.encode()),
        ("repo/vendor/lib.py", PY_CODE.encode()),
        ("repo/package-lock.json", b"{}"),
        ("repo/blob.bin", b"\x00\x01\x02" * 100),
        ("repo/huge.py", b"x = 1\n" * 50_000),
        ("repo/tiny.py", b"x = 1"),
    ]
    kept = dict(iter_repo_files(members))
    assert "repo/src/app.py" in kept
    assert "repo/README.md" in kept
    assert "repo/vendor/lib.py" not in kept
    assert "repo/blob.bin" not in kept
    assert "repo/huge.py" not in kept
    assert "repo/tiny.py" not in kept


def test_records_from_files_metadata():
    members = [("src/app.py", PY_CODE.encode()), ("PLAN.md", README.encode())]
    records = records_from_files("owner/repo", "mit", members)
    assert len(records) == 2
    by_source = {r["source"]: r for r in records}
    code = by_source["github:owner/repo/src/app.py"]
    assert code["domain"] == "code"
    assert code["license"] == "mit"
    assert code["lang"] in {"en", "de", "unknown"}
    assert len(code["id"]) == 16
    assert by_source["github:owner/repo/PLAN.md"]["license"] == "mit"
