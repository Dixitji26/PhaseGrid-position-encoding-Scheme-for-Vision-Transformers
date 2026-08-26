"""
vit_phasegrid.py

THIRD VARIANT (novel, my own idea) -- same architecture as vit_original.py
and vit_modified.py, with the positional embedding module replaced by
"PhaseGrid": a fixed 2D discrete-Fourier-transform (DFT) positional
fingerprint fed through a small LEARNED FiLM (Feature-wise Linear
Modulation; Perez et al. 2018) head that modulates attention's VALUE
vectors -- not Q/K, and not the input tokens.

This file builds the model from the SAME real timm classes as
vit_original.py / vit_modified.py (timm==1.0.28). Nothing about the
backbone is reimplemented from scratch. The only genuinely new code here
is:
  (a) the DFT fingerprint math (build_phase_grid_fingerprint), and
  (b) a small learned FiLM head (PhaseGridFiLM), and
  (c) a subclass of timm's Attention (PhaseGridAttention) that applies
      that FiLM modulation to V before the (otherwise unmodified) scaled
      dot-product attention computation.

See README.md for the exact line-by-line diff against vit_original.py and
vit_modified.py.

Why this is a genuinely different bet than RoPE (vit_modified.py), not a
variant of it: RoPE shapes attention SCORES (which patches a token looks
at) by rotating Q/K. PhaseGrid leaves attention scores purely
content-driven and instead shapes what gets AGGREGATED once attention has
already decided where to look, by modulating V. See the "Justification"
section of README.md for the full reasoning, including why I expected
this to plausibly UNDERPERFORM RoPE -- a genuinely uncertain prediction
made before training, not after seeing results.
"""

import math
import torch
import torch.nn as nn

# timm's real classes -- imported and subclassed, not reimplemented.
from timm.models.vision_transformer import VisionTransformer
from timm.layers import Attention as TimmAttention


# ---------------------------------------------------------------------------
# 2D DFT positional fingerprint (new code -- this is the actual "alternative
# scheme" for this variant)
# ---------------------------------------------------------------------------
def build_phase_grid_fingerprint(grid_size: int, num_freqs: int = None) -> torch.Tensor:
    """
    Returns a (num_patches, 4*num_freqs) tensor: for every grid position,
    sin/cos of its row and column phase at `num_freqs` distinct 2D-DFT
    frequencies. Deterministic, NOT learned -- this fixed fingerprint
    feeds into a small learned FiLM head (PhaseGridFiLM below) that turns
    it into a per-position (scale, shift) pair applied to attention's V.

    Only frequencies up to the Nyquist limit (grid_size // 2) are usable
    on a discrete grid; higher frequencies alias with lower ones
    (sin(2*pi*f*n/N) == -sin(2*pi*(N-f)*n/N)) and contribute redundant,
    not new, positional information. An earlier version of this function
    used more frequencies than this limit allows -- caught by the
    numerical check in README.md "Known pitfalls" -- so this function now
    asserts against it rather than silently wasting FiLM capacity on
    aliased signal.
    """
    nyquist = grid_size // 2
    if num_freqs is None:
        num_freqs = nyquist
    assert num_freqs <= nyquist, (
        f"num_freqs={num_freqs} exceeds Nyquist limit ({nyquist}) for grid_size={grid_size}; "
        f"frequencies beyond this alias with lower ones and add no new information."
    )

    coords = torch.arange(grid_size, dtype=torch.float32)
    rr, cc = torch.meshgrid(coords, coords, indexing="ij")
    rr, cc = rr.flatten(), cc.flatten()  # (N,)

    freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32)
    row_phase = 2 * math.pi * torch.einsum("n,f->nf", rr, freqs) / grid_size
    col_phase = 2 * math.pi * torch.einsum("n,f->nf", cc, freqs) / grid_size

    fingerprint = torch.cat(
        [row_phase.sin(), row_phase.cos(), col_phase.sin(), col_phase.cos()], dim=-1
    )  # (N, 4*num_freqs)
    return fingerprint


class PhaseGridFiLM(nn.Module):
    """
    Learned FiLM (scale+shift) generator driven by the fixed DFT
    fingerprint. Applied to V inside attention, per head, per position.
    One instance per transformer block (independent learned weights per
    layer -- position enters fresh at every layer, same as RoPE, but at
    V instead of Q/K).
    """

    def __init__(self, fingerprint_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.film = nn.Linear(fingerprint_dim, num_heads * head_dim * 2)
        # Zero-bias, small-weight init -> gamma~1, beta~0 at the start of
        # training, so the model begins close to NOT using position at
        # all and has to earn its way into using it, rather than being
        # hit with an arbitrary large perturbation to V from step one.
        nn.init.zeros_(self.film.bias)
        nn.init.trunc_normal_(self.film.weight, std=0.02)

    def forward(self, v: torch.Tensor, fingerprint: torch.Tensor) -> torch.Tensor:
        """
        v:           (B, heads, N_patches, head_dim) -- patch tokens only, no CLS
        fingerprint: (N_patches, fingerprint_dim)     -- precomputed, fixed
        """
        N = fingerprint.shape[0]
        gamma_beta = self.film(fingerprint)  # (N, heads*head_dim*2)
        gamma_beta = gamma_beta.view(N, self.num_heads, self.head_dim, 2)
        gamma, beta = gamma_beta[..., 0], gamma_beta[..., 1]  # each (N, heads, head_dim)
        gamma = 1.0 + torch.tanh(gamma)  # scale centered at 1, bounded

        gamma = gamma.permute(1, 0, 2).unsqueeze(0)  # (1, heads, N, head_dim)
        beta = beta.permute(1, 0, 2).unsqueeze(0)    # (1, heads, N, head_dim)
        return v * gamma + beta


# ---------------------------------------------------------------------------
# PhaseGrid-aware Attention: subclasses timm's real Attention class.
# Everything except the 4 lines that call phase_film() on v is timm's
# original, unmodified forward() body (see README.md for the exact diff
# against timm.layers.Attention.forward).
# ---------------------------------------------------------------------------
class PhaseGridAttention(TimmAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # phase_fingerprint / phase_film are attached post-construction by
        # build_vit_phasegrid() below, since timm's Block.forward() does
        # not thread extra positional kwargs through to attention -- this
        # keeps the diff against timm's Block class at zero lines, same
        # pattern as RoPEAttention in vit_modified.py.
        self.register_buffer("phase_fingerprint", None, persistent=False)
        self.phase_film = None  # nn.Module, attached once fingerprint_dim is known

    def forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # --- THE MODIFICATION (4 lines): FiLM-modulate patch-token V   ---
        # --- using the fixed DFT fingerprint. CLS token (index 0) is  ---
        # --- left unmodulated -- it has no spatial position by        ---
        # --- construction. Note this modifies V, NOT Q/K -- attention ---
        # --- SCORES below are computed identically to timm's original ---
        # --- and to vit_original.py; only what gets aggregated changes.---
        if self.phase_fingerprint is not None:
            v_cls, v_pat = v[:, :, :1], v[:, :, 1:]
            v_pat = self.phase_film(v_pat, self.phase_fingerprint)
            v = torch.cat([v_cls, v_pat], dim=2)
        # --- end of modification; everything below is timm's original ---
        # --- Attention.forward() manual (non-fused) attention path,   ---
        # --- byte-for-byte unchanged.                                 ---

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def build_vit_phasegrid(
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
    phase_num_freqs: int = None,
) -> nn.Module:
    """
    Build the PHASEGRID ViT: identical backbone to build_vit_original() /
    build_vit_modified(), with:
      - timm's built-in positional embedding DISABLED (pos_embed='none')
      - attn_layer=PhaseGridAttention, so every attention layer FiLM-
        modulates V using a fixed DFT positional fingerprint, instead of
        relying on an additive input embedding (original) or rotating
        Q/K (modified/RoPE).

    Every architectural knob (patch_size, embed_dim, depth, num_heads,
    mlp_ratio, dropout) is passed identically to build_vit_original() and
    build_vit_modified() -- see README.md for the parameter-parity check.
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
        pos_embed="none",               # <-- THE MODIFICATION: no additive pos embed
        drop_rate=drop_rate,
        pos_drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        attn_layer=PhaseGridAttention,  # <-- THE MODIFICATION: PhaseGrid-aware attention
    )

    # Precompute the fixed DFT fingerprint once, and attach a fresh
    # (independently-learned) FiLM head to every attention layer.
    grid_size = model.patch_embed.grid_size[0]
    head_dim = embed_dim // num_heads
    fingerprint = build_phase_grid_fingerprint(grid_size, num_freqs=phase_num_freqs)
    for block in model.blocks:
        block.attn.phase_fingerprint = fingerprint
        block.attn.phase_film = PhaseGridFiLM(fingerprint.shape[-1], num_heads, head_dim)

    return model


def count_params(model: nn.Module):
    """Returns (positional-embedding-specific params, total trainable params).
    PhaseGrid's positional-specific params are the FiLM heads' weights
    (the fixed DFT fingerprint itself is a non-learned buffer, 0 params)."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pos_specific = sum(
        p.numel() for block in model.blocks for p in block.attn.phase_film.parameters()
    )
    return pos_specific, total


if __name__ == "__main__":
    # Quick self-test: forward/backward pass + parameter count report,
    # plus the Nyquist frequency-safety check.
    torch.manual_seed(0)
    model = build_vit_phasegrid(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    y.sum().backward()
    pos_params, total_params = count_params(model)
    print(f"vit_phasegrid.py self-test: output shape {tuple(y.shape)}")
    print(f"  total trainable params:       {total_params:,}")
    print(f"  positional-embedding params:  {pos_params:,}")

    # Nyquist-safety check (see README.md "Known pitfalls").
    grid_size = 8
    nyquist = grid_size // 2
    fp = build_phase_grid_fingerprint(grid_size)
    print(f"  default fingerprint dim: {fp.shape[-1]} (expect 4*{nyquist}={4*nyquist})")
    try:
        build_phase_grid_fingerprint(grid_size, num_freqs=nyquist + 1)
        print("  ERROR: should have raised for num_freqs beyond Nyquist limit")
    except AssertionError as e:
        print(f"  Correctly rejected aliased frequency count: {e}")
