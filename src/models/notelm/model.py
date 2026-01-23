import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class NoteLMConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    max_seq_len: int = 2048
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    rotary_pct: float = 1.0
    rotary_base: int = 10000
    desc_embed_dim: int = 0
    bar_token_id: int = 0
    tie_weights: bool = True
    strict_bar_count: bool = False


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        self.cos_cached = cos
        self.sin_cached = sin
        self.max_seq_len = seq_len

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    q1, q2 = q[..., :rotary_dim], q[..., rotary_dim:]
    k1, k2 = k[..., :rotary_dim], k[..., rotary_dim:]
    q1 = (q1 * cos[..., :rotary_dim]) + (_rotate_half(q1) * sin[..., :rotary_dim])
    k1 = (k1 * cos[..., :rotary_dim]) + (_rotate_half(k1) * sin[..., :rotary_dim])
    return torch.cat([q1, q2], dim=-1), torch.cat([k1, k2], dim=-1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: NoteLMConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.rotary_dim = int(self.head_dim * config.rotary_pct)
        if self.rotary_dim % 2 == 1:
            self.rotary_dim -= 1
        self.rotary = RotaryEmbedding(self.rotary_dim, config.max_seq_len, base=config.rotary_base) if self.rotary_dim > 0 else None
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rotary is not None and self.rotary_dim > 0:
            cos, sin = self.rotary.get_cos_sin(seq_len, x.device, x.dtype)
            q, k = _apply_rotary(q, k, cos, sin, self.rotary_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~causal, torch.finfo(attn_scores.dtype).min)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_scores = attn_scores.masked_fill(attn_mask[:, None, None, :] == 0, torch.finfo(attn_scores.dtype).min)
            elif attn_mask.dim() == 3:
                if attn_mask.dtype == torch.bool:
                    attn_scores = attn_scores.masked_fill(~attn_mask[:, None, :, :], torch.finfo(attn_scores.dtype).min)
                else:
                    attn_scores = attn_scores + attn_mask[:, None, :, :]
            elif attn_mask.dim() == 4:
                if attn_mask.dtype == torch.bool:
                    attn_scores = attn_scores.masked_fill(~attn_mask, torch.finfo(attn_scores.dtype).min)
                else:
                    attn_scores = attn_scores + attn_mask
            else:
                raise ValueError("attn_mask must have 2, 3, or 4 dimensions")

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out = self.out_proj(out)
        return self.resid_dropout(out)


class FeedForward(nn.Module):
    def __init__(self, config: NoteLMConfig) -> None:
        super().__init__()
        hidden_dim = int(config.d_model * config.mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: NoteLMConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class NoteLM(nn.Module):
    def __init__(self, config: NoteLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.head.weight = self.token_embedding.weight

        if config.desc_embed_dim > 0:
            self.desc_proj = nn.Linear(config.desc_embed_dim, config.d_model, bias=False)
        else:
            self.desc_proj = None

    def _inject_desc(
        self,
        token_emb: torch.Tensor,
        ids: torch.Tensor,
        desc_embed: torch.Tensor,
    ) -> torch.Tensor:
        if self.desc_proj is None:
            raise ValueError("desc_embed was provided but desc_embed_dim is 0")

        if desc_embed.dim() != 3:
            raise ValueError("desc_embed must be (batch, num_bars, desc_dim)")

        bar_positions = ids == self.config.bar_token_id
        bar_counts = bar_positions.sum(dim=1)
        if self.config.strict_bar_count:
            if (bar_counts > desc_embed.size(1)).any():
                raise ValueError("desc_embed has fewer bars than BAR tokens")

        bar_indices = torch.cumsum(bar_positions.to(torch.int64), dim=1) - 1
        bar_indices = bar_indices.clamp(min=0, max=desc_embed.size(1) - 1)

        desc_proj = self.desc_proj(desc_embed)
        desc_at_pos = desc_proj.gather(
            1, bar_indices.unsqueeze(-1).expand(-1, -1, desc_proj.size(-1))
        )
        token_emb = token_emb + (desc_at_pos * bar_positions.unsqueeze(-1).to(token_emb.dtype))
        return token_emb

    def forward(
        self,
        ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        desc_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_emb = self.token_embedding(ids)
        if desc_embed is not None:
            token_emb = self._inject_desc(token_emb, ids, desc_embed)
        x = self.drop(token_emb)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        return self.head(x)
