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

## Filtering

The pipeline filters:
- too short / too long records
- duplicates
- non-DE/EN declared languages
- high ratio of non-Latin scripts
- unknown licenses if required

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
