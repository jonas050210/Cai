# Pilot data report

Measured on 2026-11-13 with `scripts/pilot.sh`. 18 explicitly listed,
permissively licensed GitHub repositories; nothing was scraped from code search.

Reproduce:

```bash
export GITHUB_TOKEN=...
bash scripts/pilot.sh
```

## Collection

| Repo | Records | Chars | License |
|---|---|---|---|
| fastapi/fastapi | 2,091 | 10.23M | MIT |
| pydantic/pydantic | 3,001 | 6.18M | MIT |
| gofiber/fiber | 457 | 6.10M | MIT |
| prettier/prettier | 2,962 | 5.21M | MIT |
| vitejs/vite | 1,180 | 4.55M | MIT |
| tailwindlabs/tailwindcss | 414 | 3.35M | MIT |
| Textualize/rich | 387 | 3.12M | MIT |
| pytest-dev/pytest | 401 | 2.85M | MIT |
| axios/axios | 422 | 2.40M | MIT |
| BurntSushi/ripgrep | 134 | 1.83M | Unlicense |
| python-poetry/poetry | 410 | 1.67M | MIT |
| pallets/flask | 140 | 0.93M | BSD-3-Clause |
| gin-gonic/gin | 103 | 0.73M | MIT |
| pallets/click | 90 | 0.65M | BSD-3-Clause |
| expressjs/express | 145 | 0.58M | MIT |
| encode/httpx | 54 | 0.49M | BSD-3-Clause |
| psf/requests | 47 | 0.41M | Apache-2.0 |
| chalk/chalk | 15 | 0.06M | MIT |
| **total** | **12,453** | **51.3M** | |

Licenses: MIT, Apache-2.0, BSD-3-Clause and Unlicense only. The manifest
(`data/raw/pilot.manifest.jsonl`) stores license, stars, size and the SHA-256 of
every tarball.

## Filtering (`--require-license --near-dedup`)

| Step | Records |
|---|---|
| Input | 12,453 |
| Kept | 10,438 (10,206 train / 141 val / 91 test) |
| Exact duplicates | 145 |
| Near-duplicates | 453 of 7,478 checked (6.1%) |
| Non-Latin script | 946 |
| Rejected language | 471 |
| Language by detection | 10,041 declared / 322 code-default / 75 detected |

Domain and language mix of the training split:

- code 74.1%, general 15.3%, planning 10.6% (target mixture is 65/34/1)
- English 97.4%, German 2.6%
- lengths: min 197, median 889, p90 9,995, max 119,908 chars

**Gap:** planning content is underrepresented and German is almost absent. Both
must be collected deliberately (architecture docs, ADRs, READMEs, German
documentation) before a serious run.

## Tokens

| Tokenizer | chars/token | Estimated tokens (42.2M chars) |
|---|---|---|
| 8,000 | 3.50 | 12.05M |
| 16,000 | 3.84 | 11.00M |
| 32,000 | 4.07 | 10.37M |

`codeplan-stats` measured 11.57M tokens on a 16k tokenizer sample (3.65
chars/token). Packed train split: **11.99M tokens** (code 8.90M, general 1.85M,
planning 1.24M).

## First real training step (CPU demo)

Config: 40.32M parameters, seq 256, batch 1, grad-accum 4 (1,024 tokens/step),
120 steps = 123k tokens, 2 CPU cores.

| Step | Train loss | Val loss | Val PPL |
|---|---|---|---|
| 0 | 9.77 | 9.73 | 16,883 |
| 40 | 8.04 | 7.26 | 1,417 |
| 80 | 6.98 | 7.11 | 1,229 |
| 119 | 7.28 | 7.07 | 1,172 |

Throughput on 2 CPU cores: 236-1,659 tokens/s (short sequences), and 191 tokens/s
at the real shape (seq 2048, 16x accumulation). A 4060 Ti should be 30-60x
faster — measure it with `codeplan-doctor --bench-steps 20`.

## Conclusion

The pipeline works end to end on real, licensed data and the loss falls. The
corpus is ~60x too small for a real 40M model (Chinchilla would want ~800M
tokens), so `codeplan-doctor` correctly refuses the 20,000-step run and accepts
only the 1,000-step pilot configuration.
