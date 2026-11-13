# Limitations

1. A from-scratch 50M model is small. It will not match large pretrained coding models.
2. German/English restriction is a data and prompt policy, not a hard guarantee.
3. The model can generate incorrect code, insecure code, or nonsense.
4. Reasoning traces are learned text patterns, not verified proofs.
5. The language filter is heuristic and can miss mixed-language code comments.
6. Training data quality and licensing are the user's responsibility.
7. This repo does not include a real training corpus.
8. Checkpoints are PyTorch pickle files; only load trusted checkpoints.
