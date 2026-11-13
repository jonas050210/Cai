# Limitations

1. A from-scratch 50M model is small. It will not match large pretrained coding models.
2. German/English restriction is a data and prompt policy, not a hard guarantee.
3. The model can generate incorrect code, insecure code, or nonsense.
4. Reasoning traces are learned text patterns, not verified proofs.
5. The language filter is heuristic and can miss mixed-language code comments.
   Code without prose is counted as language-neutral (`lang_source=code-default`),
   so the corpus is DE/EN + neutral code, not strictly DE/EN only.
6. `codeplan-bench` needs `--allow-exec` and runs generated code as your user with
   CPU/memory/time limits. It is not a sandbox; use a container for untrusted models.
7. The tokenizer cannot go below 274 tokens (256 bytes + 18 special tokens); asking
   for less raises an error instead of silently missing the target. Small corpora also
   may not reach the requested vocabulary size.
8. Training data quality and licensing are the user's responsibility.
9. This repo does not include a real training corpus.
10. Checkpoints are PyTorch pickle files; only load trusted checkpoints.
