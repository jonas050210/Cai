"""Decoder-only Transformer built from scratch for CodePlan-LM.

No pretrained weights are loaded. The architecture is intentionally small and
readable: RMSNorm + RoPE + SwiGLU + tied embeddings.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def precompute_rope_cache(
    seq_len: int, head_dim: int, theta: float = 10000.0, device=None, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: (B, H, T, D), cos/sin: (T, D)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_dtype, k_dtype = q.dtype, k.dtype
    q = (q.float() * cos + rotate_half(q.float()) * sin).to(q_dtype)
    k = (k.float() * cos + rotate_half(k.float()) * sin).to(k_dtype)
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        if past_kv is None and T > 1:
            if attention_mask is None:
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
                )
            else:
                mask = torch.zeros(B, 1, T, T, device=q.device, dtype=q.dtype)
                causal = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
                mask.masked_fill_(causal[None, None, :, :], float("-inf"))
                key_pad = (attention_mask == 0)[:, None, None, :]
                mask.masked_fill_(key_pad, float("-inf"))
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0, is_causal=False
                )
        else:
            # Incremental decoding: the single new token may attend to all cached tokens.
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), new_kv


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model)
        self.attention = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        h, new_kv = self.attention(
            self.input_layernorm(x), cos, sin, start_pos, use_cache, past_kv, attention_mask
        )
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class CodePlanLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cache_len = cfg.max_seq_len * 2
        cos, sin = precompute_rope_cache(cache_len, cfg.d_model // cfg.n_heads, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        use_cache: bool = False,
        past_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        B, T = input_ids.shape
        if start_pos + T > self.rope_cos.shape[0]:
            raise ValueError("Sequence exceeds RoPE cache length")

        x = self.embed_tokens(input_ids)
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        new_kvs: list = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            if self.training and self.cfg.gradient_checkpointing and not use_cache:
                x, new_kv = checkpoint(
                    layer, x, cos, sin, start_pos, use_cache, past_kv, attention_mask,
                    use_reentrant=False
                )
            else:
                x, new_kv = layer(x, cos, sin, start_pos, use_cache, past_kv, attention_mask)
            if use_cache:
                new_kvs.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_kvs

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
        return n

    def config_dict(self) -> dict:
        return asdict(self.cfg)


@torch.no_grad()
def generate(
    model: CodePlanLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Simple single-sequence generation with KV cache."""
    model.eval()
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1
    generated = input_ids.clone()

    logits, _, past_kvs = model(input_ids, use_cache=True)
    next_token_logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        logits_step = next_token_logits.clone()

        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                if logits_step[0, token_id] > 0:
                    logits_step[0, token_id] /= repetition_penalty
                else:
                    logits_step[0, token_id] *= repetition_penalty

        if temperature <= 0:
            next_token = torch.argmax(logits_step, dim=-1, keepdim=True)
        else:
            logits_step = logits_step / temperature
            if top_k > 0:
                k = min(top_k, logits_step.size(-1))
                values, _ = torch.topk(logits_step, k)
                logits_step[logits_step < values[:, [-1]]] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_step, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                mask = cumulative > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                sorted_logits[mask] = float("-inf")
                logits_step = torch.zeros_like(logits_step).scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits_step, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        logits, _, past_kvs = model(
            next_token, start_pos=generated.shape[1] - 1, use_cache=True, past_kvs=past_kvs
        )
        next_token_logits = logits[:, -1, :]

    return generated
