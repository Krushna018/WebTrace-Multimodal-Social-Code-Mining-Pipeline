from webtrace.taxonomy import classify_code_sharing


def test_repo_link_priority():
    assert classify_code_sharing("my code", "https://github.com/a/b") == "repository_link"


def test_inline_code():
    assert classify_code_sharing("try `val x = 1`") == "inline_code"


def test_none():
    assert classify_code_sharing("nice weather today") == "none"
