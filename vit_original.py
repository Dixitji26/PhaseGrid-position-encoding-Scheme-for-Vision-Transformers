"""
vit_original.py

ORIGINAL ViT -- exact positional embedding scheme from Dosovitskiy et al.,
"An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale",
ICLR 2021 (https://arxiv.org/abs/2010.11929).

This file builds the model from timm's real `VisionTransformer` implementation
(timm==1.0.28, timm.models.vision_transformer) -- it does NOT reimplement the
transformer from scratch. The only thing this file does is call timm's
VisionTransformer constructor with the config that reproduces the paper's
positional embedding: a single learned absolute positional embedding table,
added once to the patch (+ CLS) token sequence before the first transformer
block (paper Eq. 1: z0 = [x_class; x1_p*E; ...; xN_p*E] + E_pos).

See README.md for the exact line-by-line diff against vit_modified.py.
"""

import torch
import torch.nn as nn

# timm's real classes -- imported, not reimplemented.
from timm.models.vision_transformer import VisionTransformer


def build_vit_original(
    img_size: int = 32,
    patch_size: int = 4,
    in_chans: int = 3,
    num_classes: int = 10,
    embed_dim: int = 192,
    depth: int = 9,
    num_heads: int = 3,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.1,
    attn_drop_rate: float = 0.0,
) -> nn.Module:
    """
    Build the ORIGINAL ViT: timm's VisionTransformer with its default,
    paper-accurate learned absolute positional embedding (pos_embed='learn').

    Every architectural knob here (patch_size, embed_dim, depth, num_heads,
    mlp_ratio, dropout) is passed identically to build_vit_modified() in
    vit_modified.py -- see README.md for the parameter-parity check.
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        class_token=True,
        global_pool="token",
        pos_embed="learn",          # <-- THE ORIGINAL SCHEME (paper Eq. 1)
        drop_rate=drop_rate,
        pos_drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )
    return model


def count_params(model: nn.Module):
    """Returns (positional-embedding-specific params, total trainable params)."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pos_specific = model.pos_embed.numel() if model.pos_embed is not None else 0
    return pos_specific, total


if __name__ == "__main__":
    # Quick self-test: forward/backward pass + parameter count report.
    torch.manual_seed(0)
    model = build_vit_original(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    y.sum().backward()
    pos_params, total_params = count_params(model)
    print(f"vit_original.py self-test: output shape {tuple(y.shape)}")
    print(f"  total trainable params:       {total_params:,}")
    print(f"  positional-embedding params:  {pos_params:,}")
