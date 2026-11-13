from codeplan.collect import token_warning


def test_doubled_token_prefix_is_rejected():
    assert "doubled prefix" in token_warning("ghp_ghp_" + "a" * 36)


def test_truncated_classic_token_is_rejected():
    assert "40 characters" in token_warning("ghp_" + "a" * 10)


def test_plausible_tokens_pass():
    assert token_warning("ghp_" + "a" * 36) is None
    assert token_warning("github_pat_" + "a" * 80) is None
    assert token_warning(None) is None



def test_load_token_prefers_explicit_then_env(tmp_path, monkeypatch):
    from codeplan.collect import load_token

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_env")
    assert load_token("ghp_explicit") == "ghp_explicit"
    assert load_token(None) == "ghp_env"

    monkeypatch.delenv("GITHUB_TOKEN")
    fake = tmp_path / "env"
    fake.write_text('export GITHUB_TOKEN="ghp_from_file"\n')
    monkeypatch.setattr("codeplan.collect.TOKEN_FILE", fake)
    assert load_token(None) == "ghp_from_file"

    fake.write_text("# empty\n")
    assert load_token(None) is None


def test_repo_list_is_not_cut_to_default_max_repos(tmp_path, monkeypatch):
    """A 420-line list must not be silently reduced to the 200 default."""
    from codeplan import collect as collect_mod

    names = [f"owner{i}/repo{i}" for i in range(12)]
    listing = tmp_path / "repos.txt"
    listing.write_text("\n".join(names) + "\n")

    captured = {}

    def fake_collect(*args, **kwargs):
        # collect(out_path, languages, licenses, min_stars, max_repos, ...)
        captured["max_repos"] = args[4] if len(args) > 4 else kwargs.get("max_repos")
        captured.update(kwargs)
        return {"repos": 0, "records": 1, "chars": 1, "skipped": 0, "errors": 0,
                "already_done": 0, "manifest": "x"}

    monkeypatch.setattr(collect_mod, "collect", fake_collect)
    monkeypatch.setattr(collect_mod, "token_warning", lambda token: None)
    monkeypatch.setattr(
        "sys.argv",
        ["codeplan-collect", "--out", str(tmp_path / "o.jsonl"),
         "--repos-file", str(listing), "--token", "ghp_" + "a" * 36],
    )
    collect_mod.main()
    assert captured["max_repos"] == 12
