# Architecture

CodePlan-LM is a decoder-only Transformer trained from random initialization.

## Components

- Token embedding
- Transformer blocks
  - RMSNorm
  - Causal self-attention
  - RoPE positional encoding
  - SwiGLU MLP
- Final RMSNorm
- Tied language-model head

## 50M configuration

```json
{
  "vocab_size": 32000,
  "max_seq_len": 2048,
  "d_model": 512,
  "n_layers": 10,
  "n_heads": 8,
  "intermediate_size": 1408,
  "tie_embeddings": true,
  "gradient_checkpointing": true
}
```

## Why this size?

8 GB VRAM is limited. A ~50M model with bf16, checkpointing, batch size 1, and gradient accumulation is a realistic starting point. Larger models may fit only with shorter context, more offload, or smaller batch settings.

## Reasoning format

The model is not given a symbolic reasoning engine. It learns structured answer patterns from SFT data:

```text
<plan>short plan</plan>
<code>implementation</code>
<tests>tests or checks</tests>
<answer>short explanation</answer>
```

This can produce reasoning-like traces, but correctness must be evaluated with tests.
