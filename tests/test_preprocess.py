from webtrace.preprocess import extract_code_snippets, looks_like_code


def test_extract_code():
    s = "hello ```python\ndef f(x):\n return x\n``` bye"
    out = extract_code_snippets(s)
    assert len(out) == 1
    assert "def f" in out[0]


def test_looks_like_code():
    assert looks_like_code("def f(x):\n    return x;\n")
