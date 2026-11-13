# Decisions and their evidence

Every number below was produced by this repository on the pilot corpus
(18 permissively licensed repositories, 10,206 training records, 42.2 MB of text,
11.57M tokens with the 16k tokenizer). Re-run the commands to reproduce.

## 1. Vocabulary: 16k, not 32k

```
codeplan-fertility --train data/filtered/train.jsonl --vocab-sizes 8000,16000,32000
```

| vocab | tokens/char | chars/token | tokens/word | embedding params | total params | cost index |
|---|---|---|---|---|---|---|
| 8,000 | 0.286 | 3.50 | 3.14 | 4.10M | 36.21M | 10.35M |
| 16,000 | 0.261 | 3.84 | 2.86 | 8.19M | 40.31M | 10.50M |
| 32,000 | 0.246 | 4.07 | 2.69 | 16.38M | 48.50M | 11.91M |

`cost index = total params x tokens/char` (proportional to FLOPs per character of
text; see `codeplan.tokenizer_eval.recommend`). At a fixed parameter budget,
32k spends a third of the model on embeddings and is ~13% more expensive per
character than 16k for 6% better compression, so it loses. 8k is only 1.4%
cheaper than 16k but compresses noticeably worse (3.14 vs 2.86 tokens/word),
which matters for German and for long identifiers.

**Choice: 16k.** Revisit when the model moves to 150M: the embedding share falls
and larger vocabularies win back their cost. General background: embedding
matrices are ~31% of GPT-2 small but ~6.6% of Llama 3 8B, and vocabularies grow
with model width ([tokenizer costs](https://blog.ecitis.org/tokenizer-design),
[measured BPE sweep](https://dev.to/ali_82f3409ed625e8ba8515/bpe-vocab-size-23mn)).

## 2. Training tokens: ~800M-1B for a 40M model

Chinchilla-style guidance is ~20 tokens per parameter
([scaling laws](https://lazyhackers.com/article/scaling-laws),
[paper overview](https://spawn08.github.io/llmbase/06_research_papers/11-chinchilla/)).
At 40.3M parameters that is ~800M tokens. The configured 655M (20,000 steps x
32,768) is 16 tokens/parameter — close, and cheap to extend by raising
`max_steps` once the corpus allows it.

The pilot corpus is 11.6M tokens, i.e. 60x too small; `codeplan-doctor` fails the
run with `54.7 epochs` instead of letting it start. That check is intentional.

## 3. Near-duplicate removal: MinHash, 128 permutations, Jaccard 0.8

```
codeplan-prepare ... --near-dedup            # defaults: 128 perms, 32 bands, 8-byte shingles
```

On the pilot corpus this removed **453 of 7,478** checked documents (6.1%) on top
of 145 exact duplicates. The Stack reports ~40% near-duplicates for raw GitHub
data and uses 256 permutations with Jaccard 0.7-0.85
([The Stack](https://huggingface.co/datasets/bigcode/the-stack-dedup),
[The Stack v3 notes](https://huggingface.co/buckets/HuggingFaceCode/stack-v3-full)).
Our corpus is smaller and curated, so a lower rate is expected; the filter is
cheap enough to keep on by default.

Documents shorter than 500 characters are skipped: with few distinct shingles the
Jaccard estimate is too noisy on them, and they carry little token budget.

## 4. Language filter: judge code by its comments

Pure source code has no natural language, so a language detector run on the whole
file rejects it. In a controlled probe (16 records: plain Python, plain
JavaScript, German comments, French comments) the old filter dropped 12 of 16 —
including all plain code. Now:

- comments/docstrings are extracted and detected (`lang_source=detected`),
- code without prose is kept as language-neutral (`lang_source=code-default`),
- foreign-language comments are still rejected,
- an explicitly declared `de`/`en` is trusted (`lang_source=declared`).

## 5. Sequence length 2048, batch 1, grad-accum 16

32,768 tokens per optimizer step matches the >8k-32k critical-batch guidance for
small models ([batch size and scaling](https://lewislin.substack.com/p/part-8-batch-size-learning-rate-and)).
At seq 2048 the 40M model needs ~2 GB of VRAM with gradient checkpointing, so
there is room to spare on an 8 GB card. `codeplan-doctor --bench-steps` reports
the measured number; drop to 1024 only if VRAM is tight.

## 6. Learning rate 3e-4 with 500-step warmup

Published small-model recipes sit around 2e-4 with a 10x cosine decay
([ACL scaling-law study](https://aclanthology.org/2025.acl-long.1163.pdf)). 3e-4 is
slightly aggressive; it is kept because the warmup is long and the model is
small. If `loss` spikes or turns to `nan` in the first 200 steps, drop `lr` to
2e-4 before touching anything else.

## 7. bf16 + gradient checkpointing

Ada (RTX 4060 Ti) supports bf16 natively, which avoids the loss-scaling failure
modes of fp16. Checkpointing is left enabled: it costs ~30% speed and saves the
run if `seq_len` is raised to 4096 later.
