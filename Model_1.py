"""
model.py

AMC-Transformer: a transformer encoder operating directly on raw I/Q
sequences for Automatic Modulation Classification.

Design (ViT-style, adapted for 1D I/Q sequences):
    1. Patchify: split the (2, L) I/Q sequence into non-overlapping
       patches of `patch_size` consecutive time samples.
    2. Patch embedding: linearly project each flattened patch
       (2 * patch_size values) into a `d_model`-dim token embedding.
    3. Prepend a learnable [CLS] token (standard ViT/BERT trick) whose
       final representation is used for classification.
    4. Add learnable positional encodings so the model knows sample order
       (self-attention has no inherent notion of sequence position).
    5. Pass through a stack of standard transformer encoder layers
       (multi-head self-attention + feedforward, pre-norm).
    6. Classify from the [CLS] token's final representation via a small
       MLP head -> num_classes logits.

Usage:
    from model import AMCTransformer

    model = AMCTransformer(seq_len=128, num_classes=7)
    logits = model(x)  # x shape: (batch, 2, seq_len)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Learnable positional embeddings (simpler and often as effective as
    sinusoidal encoding for short sequences like ours; easy to reason about
    and debug).
    """

    def __init__(self, num_positions: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, num_positions, d_model)
        return x + self.pos_embedding


class PatchEmbedding(nn.Module):
    """Splits a (2, L) I/Q sequence into non-overlapping patches of
    `patch_size` time samples each, and linearly projects each flattened
    patch into a d_model-dim embedding.
    """

    def __init__(self, seq_len: int, patch_size: int, d_model: int, in_channels: int = 2):
        super().__init__()
        assert seq_len % patch_size == 0, (
            f"seq_len ({seq_len}) must be divisible by patch_size ({patch_size})"
        )
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        patch_dim = in_channels * patch_size
        self.proj = nn.Linear(patch_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, in_channels, seq_len) -> (batch, num_patches, patch_dim)
        batch_size, channels, seq_len = x.shape
        x = x.reshape(batch_size, channels, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, self.num_patches, channels * self.patch_size)
        return self.proj(x)  # (batch, num_patches, d_model)


class AMCTransformer(nn.Module):
    """Transformer encoder for I/Q-based modulation classification."""

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
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(seq_len, patch_size, d_model, in_channels)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = PositionalEncoding(num_patches + 1, d_model)  # +1 for CLS token
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: more stable training, standard in modern transformers
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 2, seq_len)
        batch_size = x.shape[0]

        tokens = self.patch_embed(x)  # (batch, num_patches, d_model)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch, num_patches+1, d_model)

        tokens = self.pos_encoding(tokens)
        tokens = self.dropout(tokens)

        encoded = self.encoder(tokens)  # (batch, num_patches+1, d_model)

        cls_output = self.norm(encoded[:, 0])  # take the CLS token's final representation
        logits = self.head(cls_output)  # (batch, num_classes)
        return logits

    def freeze_encoder(self, freeze: bool = True):
        """Freeze/unfreeze the patch embedding + transformer encoder layers,
        leaving the classification head trainable. Used for Stage 2/3
        transfer learning (fine-tuning on 2018.01A, then USRP captures).
        """
        for param in self.patch_embed.parameters():
            param.requires_grad = not freeze
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        self.pos_encoding.pos_embedding.requires_grad = not freeze
        self.cls_token.requires_grad = not freeze

    def unfreeze_last_n_layers(self, n: int = 2):
        """Unfreeze only the last n transformer encoder layers (partial
        fine-tuning) -- a middle ground between fully frozen and fully
        unfrozen, often gives the best transfer-learning results.
        """
        total_layers = len(self.encoder.layers)
        for i, layer in enumerate(self.encoder.layers):
            unfreeze = i >= (total_layers - n)
            for param in layer.parameters():
                param.requires_grad = unfreeze


if __name__ == "__main__":
    # Quick sanity check: verify shapes work for both dataset variants
    for seq_len in [128, 1024]:
        model = AMCTransformer(seq_len=seq_len, num_classes=7, patch_size=4 if seq_len == 128 else 16)
        dummy_input = torch.randn(8, 2, seq_len)  # batch of 8
        output = model(dummy_input)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"seq_len={seq_len}: output shape {output.shape}, params: {n_params:,}")
