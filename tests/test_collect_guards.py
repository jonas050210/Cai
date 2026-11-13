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
