from codeplan.collect import token_warning


def test_doubled_token_prefix_is_rejected():
    assert "doubled prefix" in token_warning("ghp_ghp_" + "a" * 36)


def test_truncated_classic_token_is_rejected():
    assert "40 characters" in token_warning("ghp_" + "a" * 10)


def test_plausible_tokens_pass():
    assert token_warning("ghp_" + "a" * 36) is None
    assert token_warning("github_pat_" + "a" * 80) is None
    assert token_warning(None) is None

