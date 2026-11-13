import json

from codeplan.sft_data import (
    build_pairs,
    extract_functions,
    generate,
    language_of,
)

PYTHON = '''
def parse_size(value):
    """Parse a human readable size like '12 MB' into bytes.

    Supports KB, MB and GB, case insensitive.
    """
    number, suffix = value.split()
    return int(float(number))


def test_parse_size():
    assert parse_size("1 KB") == 1024
'''

TYPESCRIPT = '''/**
 * Format a byte count for humans.
 * Uses binary units and keeps one decimal.
 */
export function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB"];
  return `${bytes} ${units[0]}`;
}
'''

GO = '''// Handler returns the HTTP handler for the API.
// It applies middleware in order.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.index)
	return mux
}
'''

RUST = '''/// Parse a duration like "12ms" into milliseconds.
/// Returns None for invalid input.
pub fn parse_duration(text: &str) -> Option<u64> {
    let trimmed = text.trim();
    trimmed.parse().ok()
}
'''


def test_language_from_extension():
    assert language_of({"source": "github:a/b/x.py"}) == "python"
    assert language_of({"source": "github:a/b/x.tsx"}) == "typescript"
    assert language_of({"source": "github:a/b/x.js"}) == "javascript"
    assert language_of({"source": "github:a/b/README.md"}) is None
    assert language_of({"language": "ts", "source": "x"}) == "typescript"


def test_python_function_with_docstring_and_test():
    functions = extract_functions(PYTHON, language="python")
    assert [f.name for f in functions] == ["parse_size"]
    assert "human readable size" in functions[0].description
    assert "test_parse_size" in functions[0].tests


def test_brace_languages():
    ts = extract_functions(TYPESCRIPT, language="typescript")
    assert [f.name for f in ts] == ["formatBytes"]
    assert ts[0].code.rstrip().endswith("}")

    go = extract_functions(GO, language="go")
    assert [f.name for f in go] == ["Handler"]
    assert "mux.HandleFunc" in go[0].code

    rust = extract_functions(RUST, language="rust")
    assert [f.name for f in rust] == ["parse_duration"]
    assert "trimmed.parse()" in rust[0].code


def test_undocumented_functions_are_skipped():
    code = "def helper(a, b):\n    return a + b\n"
    assert extract_functions(code, language="python") == []


def test_pairs_cover_implement_and_explain():
    function = extract_functions(PYTHON, language="python")[0]
    kinds = [pair["kind"] for pair in build_pairs(function)]
    assert kinds == ["implement", "explain", "test"]
    implement = build_pairs(function)[0]
    assert "parse_size" in implement["user"]
    assert implement["answer"].startswith("def parse_size")


def test_generate_writes_train_and_val(tmp_path):
    corpus = tmp_path / "train.jsonl"
    records = [
        {"text": PYTHON, "source": "github:a/b/util.py"},
        {"text": TYPESCRIPT, "source": "github:a/b/util.ts"},
        {"text": "# only a readme\n\ntext", "source": "github:a/b/README.md"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    report = generate(corpus, tmp_path / "sft", min_description=10)
    assert report["pairs"] >= 3
    assert report["per_language"]["python"] >= 1
    assert report["per_language"]["typescript"] >= 1
    assert report["skipped_files"] == 1
    train = (tmp_path / "sft" / "train.jsonl").read_text().splitlines()
    assert train
    first = json.loads(train[0])
    assert first["system"] and first["user"] and first["answer"]
    assert (tmp_path / "sft" / "val.jsonl").exists()


def test_generate_respects_max_pairs(tmp_path):
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps({"text": PYTHON, "source": "a/util.py"}) + "\n")
    report = generate(corpus, tmp_path / "sft", max_pairs=1, min_description=10)
    assert report["pairs"] == 1
