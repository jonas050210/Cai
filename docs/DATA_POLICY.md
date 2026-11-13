# Data Policy

## Allowed natural languages

Only German and English natural-language text should be used.

Programming-language syntax is allowed: Python, JavaScript, TypeScript, C, C++, Rust, Go, SQL, etc. But comments, docstrings, commit messages, and documentation should be DE/EN.

## Target mixture

Pretraining target:
- 65% code
- 34% planning / architecture / specs / roadmaps
- 1% general German/English language

This is enforced approximately by domain-weighted sampling from packed binary files.

## Required metadata

Each record should have:
- `text`
- `lang`: `de` or `en`
- `domain`: `code`, `planning`, or `general`
- `license`
- `source`

Use `--require-license` for serious runs.

## Collection

`codeplan-collect` only accepts repositories whose GitHub license identifier is on
an explicit whitelist (MIT, Apache-2.0, BSD-2/3-Clause, ISC, CC0-1.0, Unlicense,
MPL-2.0). It downloads tarballs (no git history), skips vendored/build/test
directories and binary or generated files, and writes a manifest next to the JSONL
with license, stars, size and the SHA-256 of every tarball. Every record carries
its license and a `github:owner/repo/path` source.

## Filtering

The pipeline filters:
- too short / too long records
- exact duplicates (normalized SHA-256)
- non-DE/EN languages
- high ratio of non-Latin scripts
- unknown licenses if required

For `domain=code` only comments, docstrings and similar prose are language-checked
(`lang_source=detected`). Code without prose is kept as language-neutral
(`lang_source=code-default`) instead of being dropped. Declared `de`/`en` is trusted
(`lang_source=declared`).

This is not a perfect language detector. Manual audit is still needed.

## Do not commit

Do not commit:
- raw datasets
- filtered datasets
- packed `.bin` files
- tokenizer artifacts from real data
- checkpoints
- logs containing private data

The `.gitignore` excludes these by default.
