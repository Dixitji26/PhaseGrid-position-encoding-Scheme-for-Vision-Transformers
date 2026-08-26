"""
vit_modified.py

MODIFIED ViT -- same architecture as vit_original.py, with the positional
embedding module replaced by 2D Axial Rotary Position Embedding (RoPE).

This file builds the model from the SAME real timm classes as
vit_original.py (timm==1.0.28, timm.models.vision_transformer /
timm.layers). Nothing about the backbone is reimplemented from scratch:
patch embedding, MLP, LayerNorm, block structure, and the overall
VisionTransformer assembly are all timm's real code. The only genuinely
new code in this file is:

  (a) the RoPE math itself (build_2d_axial_freqs / apply_rope), and
  (b) a ~15-line subclass of timm's Attention that calls that RoPE math
      on Q/K before the (otherwise completely unmodified) scaled
      dot-product attention computation.

See README.md for the exact line-by-line diff against vit_original.py --
every line that differs between the two files is listed there with line
numbers from THIS file and from vit_original.py.

Justification (see notebook Section 3 / README "Justification" section for
the full version): the original scheme adds a single learned absolute
position vector once, at the input, and that signal must survive every
residual block as a static bias to still matter for attention deep in the
network. RoPE instead rotates Q/K by a position-dependent angle inside
EVERY attention layer, which makes the attention score between two tokens
a function of their RELATIVE (row, col) offset by mathematical
construction, not something the model has to discover from a lookup
table. It is also fully parameter-free (0 extra learned weights vs.
12,480 for the learned table).
"""

import math
import torch
import torch.nn as nn

# timm's real classes -- imported and subclassed, not reimplemented.
from timm.models.vision_transformer import VisionTransformer
from timm.layers import Attention as TimmAttention


# ---------------------------------------------------------------------------
# 2D Axial RoPE math (new code -- this is the actual "alternative scheme")
# ---------------------------------------------------------------------------
def build_2d_axial_freqs(grid_size: int, head_dim: int, base: float = 100.0):
    """
    Precompute rotary sin/cos tables for a square patch grid.

    head_dim is split in half: the first half of channels encodes ROW
    position, the second half encodes COLUMN position (axial
    factorization). Each axis-block is rotated with its OWN independent
    split-half rotation (see apply_rope below) -- naively concatenating
    [row,row,col,col] angles and applying a single GLOBAL rotate_half
    across the whole head_dim mixes the row block's second half with the
    column block's first half and destroys the relative-position property
    that is RoPE's entire point. This was caught and fixed by the
    numerical check in the accompanying notebook (Section 2) before any
    model was trained -- see README.md "Known pitfalls" section.

    Only frequencies up to the Nyquist limit (grid_size // 2) are usable
    on a discrete grid; higher frequencies alias with lower ones
    (sin(2*pi*f*n/N) == -sin(2*pi*(N-f)*n/N)) and add no new information.
    base=100.0 keeps all used frequencies comfortably below that limit
    for the grid sizes used in this assignment (grid_size=8).
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D axial RoPE"
    dim_half = head_dim // 2
    freq_len = dim_half // 2

    inv_freq = 1.0 / (base ** (torch.arange(0, freq_len, dtype=torch.float32) / freq_len))

    coords = torch.arange(grid_size, dtype=torch.float32)
    row_ids = coords.view(grid_size, 1).expand(grid_size, grid_size).reshape(-1)
    col_ids = coords.view(1, grid_size).expand(grid_size, grid_size).reshape(-1)

    row_angles = torch.einsum("n,f->nf", row_ids, inv_freq)
    col_angles = torch.einsum("n,f->nf", col_ids, inv_freq)

    angles = torch.cat([row_angles, row_angles, col_angles, col_angles], dim=-1)
    return angles.cos(), angles.sin()


def _rotate_half_local(x: torch.Tensor) -> torch.Tensor:
    """Local split-half rotation applied WITHIN one axis-block."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, heads, N, head_dim); cos/sin: (N, head_dim) from build_2d_axial_freqs.

    Rotates the row-block (first half of head_dim) and col-block (second
    half) INDEPENDENTLY, each with its own local rotate_half. This keeps
    the two axes from mixing and preserves exact relative-position
    invariance: <RoPE(q, pos_i), RoPE(k, pos_j)> depends only on
    (row_i - row_j, col_i - col_j), not on absolute (row_i, col_i).
    Verified numerically in the accompanying notebook, Section 2.
    """
    d = x.shape[-1]
    dh = d // 2
    x_row, x_col = x[..., :dh], x[..., dh:]
    cos_row, cos_col = cos[..., :dh], cos[..., dh:]
    sin_row, sin_col = sin[..., :dh], sin[..., dh:]

    x_row_rot = x_row * cos_row + _rotate_half_local(x_row) * sin_row
    x_col_rot = x_col * cos_col + _rotate_half_local(x_col) * sin_col
    return torch.cat([x_row_rot, x_col_rot], dim=-1)


# ---------------------------------------------------------------------------
# RoPE-aware Attention: subclasses timm's real Attention class.
# Everything except the 5 lines that call apply_rope() on q/k is timm's
# original, unmodified forward() body (see README.md for the exact diff
# against timm.layers.Attention.forward).
# ---------------------------------------------------------------------------
class RoPEAttention(TimmAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # rope_cos / rope_sin are attached post-construction by
        # build_vit_modified() below, since timm's Block.forward() does
        # not thread extra positional kwargs through to attention -- this
        # keeps the diff against timm's Block class at zero lines.
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # --- THE MODIFICATION (5 lines): rotate patch-token Q/K by 2D  ---
        # --- axial RoPE. CLS token (index 0) is left un-rotated -- it  ---
        # --- has no spatial position by construction.                 ---
        if self.rope_cos is not None:
            q_cls, q_pat = q[:, :, :1], q[:, :, 1:]
            k_cls, k_pat = k[:, :, :1], k[:, :, 1:]
            q_pat = apply_rope(q_pat, self.rope_cos, self.rope_sin)
            k_pat = apply_rope(k_pat, self.rope_cos, self.rope_sin)
            q = torch.cat([q_cls, q_pat], dim=2)
            k = torch.cat([k_cls, k_pat], dim=2)
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


def build_vit_modified(
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
    rope_base: float = 100.0,
) -> nn.Module:
    """
    Build the MODIFIED ViT: identical backbone to build_vit_original() in
    vit_original.py, with:
      - timm's built-in positional embedding DISABLED (pos_embed='none')
      - attn_layer=RoPEAttention, so every attention layer rotates Q/K
        with 2D axial RoPE instead of relying on an additive input embedding.

    Every architectural knob (patch_size, embed_dim, depth, num_heads,
    mlp_ratio, dropout) is passed identically to build_vit_original() --
    see README.md for the parameter-parity check.
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
        pos_embed="none",           # <-- THE MODIFICATION: no additive pos embed
        drop_rate=drop_rate,
        pos_drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        attn_layer=RoPEAttention,   # <-- THE MODIFICATION: RoPE-aware attention
    )

    # Precompute RoPE tables once and attach to every attention layer.
    grid_size = model.patch_embed.grid_size[0]
    head_dim = embed_dim // num_heads
    cos, sin = build_2d_axial_freqs(grid_size, head_dim, base=rope_base)
    for block in model.blocks:
        block.attn.rope_cos = cos
        block.attn.rope_sin = sin

    return model


def count_params(model: nn.Module):
    """Returns (positional-embedding-specific params, total trainable params).
    RoPE is parameter-free, so positional-specific params is always 0."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0, total


if __name__ == "__main__":
    # Quick self-test: forward/backward pass + parameter count report,
    # plus the RoPE relative-position invariance check.
    torch.manual_seed(0)
    model = build_vit_modified(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    y.sum().backward()
    pos_params, total_params = count_params(model)
    print(f"vit_modified.py self-test: output shape {tuple(y.shape)}")
    print(f"  total trainable params:       {total_params:,}")
    print(f"  positional-embedding params:  {pos_params:,}")

    # Relative-position invariance check (see README.md "Known pitfalls").
    grid, head_dim = 8, 64
    cos, sin = build_2d_axial_freqs(grid, head_dim, base=100.0)

    def idx(r, c):
        return r * grid + c

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def rotated_dot(rq, cq, rk, ck):
        i, j = idx(rq, cq), idx(rk, ck)
        qr = apply_rope(q, cos[i : i + 1], sin[i : i + 1])
        kr = apply_rope(k, cos[j : j + 1], sin[j : j + 1])
        return (qr * kr).sum().item()

    d_a = rotated_dot(1, 1, 2, 3)
    d_b = rotated_dot(4, 4, 5, 6)
    print(f"  RoPE relative-position check: same offset at two anchors -> "
          f"{d_a:.6f} vs {d_b:.6f} (match: {abs(d_a - d_b) < 1e-4})")
