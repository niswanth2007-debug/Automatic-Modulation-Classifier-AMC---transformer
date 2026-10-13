"""
model_v2.py

Iteration 2 of AMCTransformer. Adds three targeted, low-risk enhancements
informed by recent AMC literature, while staying strictly in raw-I/Q
space (no image conversion, no new external dependencies):

1. Rotary Position Embedding (RoPE) -- informed by MoEformer (Wang et al.).
   Replaces the learnable absolute positional encoding with a relative,
   rotation-based scheme. Physically motivated for radio signals: I/Q
   phase evolves as a rotation over time, so encoding *relative* position
   via rotation is a more natural match than an arbitrary learned vector
   per absolute position. Also removes the seq-len-dependent positional
   embedding that had to be discarded during Stage 1 -> Stage 2 transfer
   (RoPE is computed on the fly, not stored as length-specific weights).

2. Cross-layer feature fusion -- informed by MSFF-ViT (Wu et al.).
   Instead of classifying from only the final layer's CLS output, we
   concatenate CLS representations from an early, middle, and late layer.
   MSFF-ViT's own ablation reports a modest but real accuracy gain from
   this (+0.66% over plain ViT in their setup).

3. Attention pooling -- informed by MoEformer (Wang et al.).
   Instead of relying solely on the CLS token, an attention-pooling head
   also learns to weight and combine ALL patch tokens (not just CLS),
   letting the model learn which time segments are most discriminative
   rather than uniformly averaging or relying on one summary token.

Explicitly NOT included (see project notes for reasoning): MHSSA subspace
attention, KAN classifier, multi-modal image inputs (constellation/STFT/
power spectrum), Mixture-of-Experts routing, TDFF/CPCA fusion modules.
These require external dependencies, custom attention kernels, or an
image-based pipeline already rejected earlier in the project for good,
documented reasons (see conversation notes on 1D-vs-image comparisons).

Usage:
    from model_v2 import AMCTransformerV2
    model = AMCTransformerV2(seq_len=128, num_classes=7)
    logits = model(x)  # x shape: (batch, 2, seq_len)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def build_rope_cache(seq_len: int, dim: int, base: float = 10000.0, device=None):
    """Precompute the cos/sin rotation angles for RoPE.

    dim must be even (RoPE rotates pairs of dimensions). Returns
    (cos, sin) each of shape (seq_len, dim).
    """
    assert dim % 2 == 0, "RoPE dimension must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input, the standard RoPE helper."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applies rotary embeddings to query and key tensors.

    q, k shape: (batch, heads, seq_len, head_dim)
    cos, sin shape: (seq_len, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to Q/K, replacing the
    positional-encoding-then-standard-attention approach. Note: the CLS
    token (position 0) also gets rotated -- this is standard practice
    (e.g. ViT/BERT variants using RoPE treat CLS as position 0) and works
    fine in practice since RoPE encodes relative offsets between all
    token pairs, including CLS-to-patch relationships.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        q, k = apply_rope(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class RoPETransformerLayer(nn.Module):
    """Pre-norm transformer encoder layer using RoPE self-attention
    instead of standard nn.MultiheadAttention with additive positional
    encoding.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch embedding (unchanged from v1)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, seq_len: int, patch_size: int, d_model: int, in_channels: int = 2):
        super().__init__()
        assert seq_len % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        patch_dim = in_channels * patch_size
        self.proj = nn.Linear(patch_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, seq_len = x.shape
        x = x.reshape(batch_size, channels, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, self.num_patches, channels * self.patch_size)
        return self.proj(x)


# ---------------------------------------------------------------------------
# 3. Attention pooling
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Learns a single query vector that attends over all tokens (patches
    + CLS) to produce a pooled summary, instead of relying only on the
    CLS token's final state. Lets the model learn which time segments
    matter most for classification.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, num_tokens, d_model)
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.attn(query, x, x)  # (batch, 1, d_model)
        return pooled.squeeze(1)


# ---------------------------------------------------------------------------
# Full model: AMCTransformerV2
# ---------------------------------------------------------------------------

class AMCTransformerV2(nn.Module):
    """Iteration 2: RoPE + cross-layer fusion + attention pooling."""

    def __init__(
        self,
        seq_len: int = 128,
        num_classes: int = 7,
        patch_size: int = 4,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 256,
        dropout: float = 0.1,
        in_channels: int = 2,
        fusion_layer_indices=(1, 3, 5),  # shallow, middle, deep (0-indexed, out of 6 layers)
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(seq_len, patch_size, d_model, in_channels)
        num_patches = self.patch_embed.num_patches
        num_tokens = num_patches + 1  # +1 for CLS

        assert (d_model // num_heads) % 2 == 0, "head_dim must be even for RoPE"

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            RoPETransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fusion_layer_indices = fusion_layer_indices

        head_dim = d_model // num_heads
        cos, sin = build_rope_cache(num_tokens, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPooling(d_model)

        fused_dim = d_model * (len(fusion_layer_indices) + 1)  # + final layer output
        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.dropout(tokens)

        cos = self.rope_cos.to(tokens.device)
        sin = self.rope_sin.to(tokens.device)

        collected_cls = []
        for i, layer in enumerate(self.layers):
            tokens = layer(tokens, cos, sin)
            if i in self.fusion_layer_indices:
                collected_cls.append(tokens[:, 0])  # CLS token at this depth

        tokens = self.norm(tokens)
        pooled = self.pool(tokens)  # attention-pooled summary over all tokens

        fused = torch.cat(collected_cls + [pooled], dim=-1)
        logits = self.head(fused)
        return logits

    def freeze_encoder(self, freeze: bool = True):
        for param in self.patch_embed.parameters():
            param.requires_grad = not freeze
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = not freeze
        self.cls_token.requires_grad = not freeze

    def unfreeze_last_n_layers(self, n: int = 2):
        total_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            unfreeze = i >= (total_layers - n)
            for param in layer.parameters():
                param.requires_grad = unfreeze


if __name__ == "__main__":
    # Smoke test: verify shapes work for both dataset variants, and that
    # RoPE cache regenerates correctly for a different seq_len.
    for seq_len, patch_size in [(128, 4), (1024, 16)]:
        model = AMCTransformerV2(seq_len=seq_len, num_classes=7, patch_size=patch_size)
        dummy_input = torch.randn(8, 2, seq_len)
        output = model(dummy_input)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"seq_len={seq_len}: output shape {output.shape}, params: {n_params:,}")
