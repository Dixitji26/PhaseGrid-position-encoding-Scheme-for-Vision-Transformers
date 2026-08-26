# ViT Positional Embedding Comparison — README

This submission compares **three** ViT variants that are **architecturally identical** except for
how positional information reaches the model. All three are built from **timm's real
`VisionTransformer` implementation** (`timm==1.0.28`, `timm.models.vision_transformer` /
`timm.layers`) — none of the three models is implemented from scratch.

The assignment asks for a two-way comparison (original vs. one modified scheme); this submission
includes a third, additional variant (`vit_phasegrid.py`) as an extra, self-directed exploration of
a second, differently-motivated alternative scheme. The required two-way comparison is fully
covered by `vit_original.py` vs. `vit_modified.py` — `vit_phasegrid.py` is bonus material and can be
ignored if only the required two-way comparison is being graded.

| File | Purpose |
|---|---|
| `vit_original.py` | Original ViT — timm's built-in learned absolute positional embedding (paper Eq. 1) |
| `vit_modified.py` | **Required modified variant** — positional embedding replaced with 2D Axial RoPE |
| `vit_phasegrid.py` | *(Additional, not required)* — positional embedding replaced with a novel DFT-fingerprint → FiLM-on-V scheme |
| `train.py` | Shared training script; `--variant original\|modified\|phasegrid` selects the model, everything else is identical |
| `compare_results.py` | Reads `results_*.json` outputs, prints the accuracy table, saves the combined loss plot (works with 2 or 3 variants) |
| `make_figures.py` | Generates IEEE-quality figures for a report: loss curves, validation accuracy, and test-set prediction-confidence histogram (see "Generating report figures" below) |

## How to reproduce

```bash
pip install torch torchvision timm matplotlib

python train.py --variant original --epochs 60
python train.py --variant modified --epochs 60

# optional third variant:
python train.py --variant phasegrid --epochs 60

python compare_results.py
```

## Generating report figures

After training, run:

```bash
python make_figures.py
```

This produces three IEEE-conference-style figures (300 DPI PNG + vector PDF) in `./figures/`:

- `loss_curves.{png,pdf}` — training loss (opaque) + validation loss (faded), all trained variants overlaid
- `val_accuracy.{png,pdf}` — validation accuracy per epoch, with each variant's final test accuracy marked as a dotted horizontal line
- `confidence_histogram.{png,pdf}` — distribution of per-image prediction confidence (max softmax probability) on the CIFAR-10 test set, all trained variants overlaid

The first two figures only need the saved `results_*.json` files. The confidence histogram
additionally reloads each `checkpoint_*.pt` and re-runs inference over the test set (fast — a
single evaluation pass, not another training run). `make_figures.py` works with however many
variants you've trained (1, 2, or 3) — it skips anything missing rather than failing.

Figures use a serif font, a colorblind-safe palette (Okabe-Ito) with a distinct line style per
variant (so they remain distinguishable in grayscale printing), and embedded TrueType fonts in the
PDF output, matching typical IEEE camera-ready formatting requirements.

Dataset: **CIFAR-10** (`torchvision.datasets.CIFAR10`) — 10 classes, 32×32 RGB images, 50,000
train / 10,000 test images (the last 5,000 training images are held out as a validation split for
checkpoint selection, since CIFAR-10 ships with only train/test). Auto-downloads via
`download=True`. CIFAR-10 is one of the benchmark datasets explicitly reported in Dosovitskiy et
al. 2021 (Table 2, Table 5), satisfying the assignment's "choose any one benchmark dataset reported
in the paper" requirement. Trained from scratch, no pretrained weights, as required.

---

## Exact line-by-line diff: `vit_original.py` vs. `vit_modified.py`

Every line below is a genuine functional difference between the two files — docstring wording
differences are omitted since they don't affect model behavior. Line numbers refer to the current
version of each file.

### 1. Imports — one extra import in the modified file

| `vit_original.py` | `vit_modified.py` |
|---|---|
| L19–23: `import torch`, `import torch.nn as nn`, `from timm.models.vision_transformer import VisionTransformer` | L35–41: same three imports, **plus** `import math` and `from timm.layers import Attention as TimmAttention` |

The extra import (`TimmAttention`) is needed because the modification subclasses timm's real
`Attention` class — see item 3 below.

### 2. New code: the RoPE math itself (`vit_modified.py` only, no equivalent in `vit_original.py`)

| Lines | What |
|---|---|
| `vit_modified.py` L47–82 | `build_2d_axial_freqs()` — precomputes the rotary sin/cos tables for the patch grid |
| `vit_modified.py` L85–89 | `_rotate_half_local()` — helper for the axis-independent rotation |
| `vit_modified.py` L92–111 | `apply_rope()` — applies the rotation to Q or K |

This is genuinely new code (not present in timm), because RoPE is not one of timm's built-in
positional embedding options. It is the actual "alternative scheme" this assignment asks for.

### 3. New code: `RoPEAttention` class (`vit_modified.py` only)

`vit_modified.py` L120–160 defines `class RoPEAttention(TimmAttention)`, a subclass of timm's real
`Attention` (`timm.layers.Attention`). There is no equivalent class in `vit_original.py` — the
original variant uses timm's `Attention` directly, unmodified.

Comparing `RoPEAttention.forward` (`vit_modified.py` L130–160) against timm's own
`Attention.forward` (`timm/layers/attention.py`, imported not vendored):

- **Identical to timm's original:** the QKV projection/reshape (L132), `q,k,v = qkv.unbind(0)`
  (L133), `q_norm`/`k_norm` (L134), the scale-then-matmul-then-softmax attention computation
  (L150–154), and the output reshape/norm/proj/drop (L156–159) are all byte-for-byte the same
  operations as timm's manual (non-fused) attention path.
- **New (the actual modification):** `vit_modified.py` L139–145, 7 lines, guarded by
  `if self.rope_cos is not None:` — rotates the patch-token Q/K via `apply_rope()` before the
  attention score computation. The CLS token (index 0) is left un-rotated.
- **Intentionally omitted vs. timm's original:** timm's `Attention.forward` also supports a fused
  `scaled_dot_product_attention` fast path and an optional sigmoid `gate`. `RoPEAttention.forward`
  implements only the manual (non-fused, non-gated) path, since injecting RoPE into Q/K requires
  a materialized attention-score matrix — the same manual path timm falls back to whenever
  `qk_norm`, `scale_norm`, or non-default attention masking is used. Both variants are configured
  identically here (`qk_norm=False`, `scale_norm=False`, `gated=False` — timm's defaults), so this
  does not introduce any asymmetry between the two models being compared.

### 4. The constructor call — the actual positional-embedding switch

| `vit_original.py` (`build_vit_original`, L46–62) | `vit_modified.py` (`build_vit_modified`, L187–204) |
|---|---|
| L58: `pos_embed="learn",` | L199: `pos_embed="none",` |
| *(nothing)* | L203: `attn_layer=RoPEAttention,` |
| *(nothing — the rest of the call is identical: `img_size`, `patch_size`, `in_chans`, `num_classes`, `embed_dim`, `depth`, `num_heads`, `mlp_ratio`, `qkv_bias=True`, `class_token=True`, `global_pool="token"`, `drop_rate`, `pos_drop_rate`, `attn_drop_rate` — all passed with the same values)* | *(same)* |

This is the entire architectural change: **one flag flipped** (`pos_embed="learn"` →
`pos_embed="none"`, disabling timm's built-in additive positional table) **and one flag added**
(`attn_layer=RoPEAttention`, routing every attention layer through the RoPE-aware subclass instead
of timm's default `Attention`). Every other constructor argument — and therefore every other part
of the architecture (patch embedding, depth, width, heads, MLP ratio, dropout, class token,
pooling) — is passed identically between the two functions.

### 5. New code: attaching RoPE tables to the model (`vit_modified.py` only)

`vit_modified.py` L206–212, inside `build_vit_modified()`, after the `VisionTransformer(...)` call:

```python
grid_size = model.patch_embed.grid_size[0]
head_dim = embed_dim // num_heads
cos, sin = build_2d_axial_freqs(grid_size, head_dim, base=rope_base)
for block in model.blocks:
    block.attn.rope_cos = cos
    block.attn.rope_sin = sin
```

No equivalent code exists in `vit_original.py`. This precomputes the RoPE sin/cos tables once and
attaches them to every attention layer's buffers. It is necessary because timm's `Block.forward()`
does not thread extra positional kwargs through to `Attention.forward()` — attaching the tables as
buffers avoids needing to also subclass/modify timm's `Block` class, keeping the diff against
timm's block-level code at **zero lines**.

### 6. `count_params()` — differs by construction, not by choice

| `vit_original.py` L66–70 | `vit_modified.py` L217–221 |
|---|---|
| Returns `model.pos_embed.numel()` as the positional-parameter count | Always returns `0`, since RoPE has no learned parameters |

---

## Exact line-by-line diff: `vit_original.py` vs. `vit_phasegrid.py` *(additional variant)*

Same format as the RoPE diff above. `vit_phasegrid.py` follows the identical pattern — subclass
timm's `Attention`, disable the built-in positional table, route through the subclass via
`attn_layer=` — but modulates **V** rather than rotating **Q/K**.

### 1. Imports — same extra imports as `vit_modified.py`

`vit_phasegrid.py` L34–40 adds `import math` and `from timm.layers import Attention as
TimmAttention`, same as `vit_modified.py`, for the same reason (subclassing timm's real
`Attention`).

### 2. New code: the DFT fingerprint + FiLM math (`vit_phasegrid.py` only)

| Lines | What |
|---|---|
| `vit_phasegrid.py` L47–82 | `build_phase_grid_fingerprint()` — precomputes the fixed 2D-DFT sin/cos fingerprint per grid position (Nyquist guard at L67) |
| `vit_phasegrid.py` L86–125 | `PhaseGridFiLM` — a small **learned** `nn.Linear` that turns the fixed fingerprint into a per-position (scale, shift) pair (gentle zero-bias init at L104) |

Unlike RoPE's fingerprint (fixed function of position, no learned parameters at all), PhaseGrid's
fingerprint is fixed but the **use** of it is learned: `PhaseGridFiLM.film` is a real `nn.Linear`
with trainable weights. This is a deliberate design difference — see "Justification" below.

### 3. New code: `PhaseGridAttention` class (`vit_phasegrid.py` only)

`vit_phasegrid.py` L129 defines `class PhaseGridAttention(TimmAttention)`, with `forward()`
starting at L140. Comparing against timm's original `Attention.forward`:

- **Identical to timm's original:** QKV projection/reshape, `q,k,v = qkv.unbind(0)`, `q_norm`/
  `k_norm`, the scale-then-matmul-then-softmax attention computation, and the output reshape/norm/
  proj/drop — same as `vit_modified.py`, unchanged from timm.
- **New (the actual modification):** `vit_phasegrid.py` L152–155, 4 lines, guarded by
  `if self.phase_fingerprint is not None:` — FiLM-modulates the patch-token **V** via
  `self.phase_film(v_pat, self.phase_fingerprint)` before the attention computation. The CLS token
  is left unmodulated.
- **Critical distinction from RoPE:** this modification touches **V**, not Q/K. The line
  `attn = q @ k.transpose(-2, -1)` (L161) — the actual attention-score computation, i.e. *which*
  patches attend to *which* — is completely untouched, identical to `vit_original.py`'s attention
  scores. Only what gets aggregated once attention weights are already computed changes.

### 4. The constructor call

| `vit_original.py` L58 | `vit_phasegrid.py` (`build_vit_phasegrid`) |
|---|---|
| `pos_embed="learn",` | `pos_embed="none",` (L211) |
| *(nothing)* | `attn_layer=PhaseGridAttention,` (L215) |

Same pattern as the RoPE diff: one flag flipped, one flag added, every other constructor argument
identical.

### 5. New code: attaching fingerprint + FiLM heads to the model (`vit_phasegrid.py` only)

`vit_phasegrid.py` L220–225, inside `build_vit_phasegrid()`:

```python
grid_size = model.patch_embed.grid_size[0]
head_dim = embed_dim // num_heads
fingerprint = build_phase_grid_fingerprint(grid_size, num_freqs=phase_num_freqs)
for block in model.blocks:
    block.attn.phase_fingerprint = fingerprint
    block.attn.phase_film = PhaseGridFiLM(fingerprint.shape[-1], num_heads, head_dim)
```

Same rationale as RoPE's table-attachment code: avoids modifying timm's `Block` class. Note each
block gets its **own** `PhaseGridFiLM` instance (independently learned weights per layer) — position
is re-learned fresh at every layer, same principle as RoPE, but via a learned gate rather than a
fixed rotation.

---

## Parameter-parity verification

Confirmed by running each file's `if __name__ == "__main__"` self-test (`python vit_original.py` /
`python vit_modified.py` / `python vit_phasegrid.py`) at `img_size=32, patch_size=4,
embed_dim=192, depth=9, num_heads=3`:

| | Total params | Positional-specific params | Backbone params |
|---|---|---|---|
| Original | 4,028,170 | 12,480 | 4,015,690 |
| Modified (RoPE) | 4,015,690 | 0 | 4,015,690 |
| PhaseGrid (additional) | 4,074,442 | 58,752 | 4,015,690 |

**Backbone parameter count matches exactly across all three** (4,015,690 in every row) — every
model differs from the others only by its positional-mechanism parameters, and nothing else in the
architecture. This is the concrete evidence that "everything else is identical," as required by the
assignment. (PhaseGrid's 58,752 positional-specific params come from nine independent
`PhaseGridFiLM` heads — one per transformer block — each a `Linear(16, 384)`: `16×384 + 384 = 6,528`
params per block × 9 blocks = 58,752.)

---

## Known pitfalls caught during development (see code comments for details)

1. **RoPE axis-mixing bug.** A naive first implementation concatenated row/column rotation angles
   as `[row, row, col, col]` and applied one *global* `rotate_half` across the whole `head_dim`.
   This silently mixes the row axis's second half with the column axis's first half during
   rotation, which destroys the relative-position invariance property that is the entire reason to
   use RoPE. Fixed in `apply_rope()` (`vit_modified.py` L92–111) by rotating the row-block and
   column-block **independently**, each with its own local split-half rotation. Verified
   numerically: the dot product between a rotated query and rotated key at the same *relative*
   offset (e.g. (Δrow, Δcol) = (+1, +2)) but two different *absolute* anchor positions produces
   identical values only after this fix.
2. **Frequency aliasing (PhaseGrid).** On a discrete 8×8 patch grid, DFT frequencies above the
   Nyquist limit (`grid_size // 2 = 4`) alias with lower frequencies
   (`sin(2πfn/N) == -sin(2π(N-f)n/N)`), contributing redundant rather than new positional
   information. An earlier version of `build_phase_grid_fingerprint` (`vit_phasegrid.py`) used more
   frequencies than this limit allows, silently wasting FiLM capacity on aliased, duplicate signal
   instead of genuinely new positional detail. Fixed by asserting `num_freqs <= grid_size // 2` in
   `build_phase_grid_fingerprint` (`vit_phasegrid.py` L68–72) — verified by
   `python vit_phasegrid.py`, which confirms both the default frequency count and that requesting
   an aliased frequency count correctly raises.

---

## Justification: why 2D Axial RoPE?

The original ViT adds a single *learned, absolute* positional embedding vector to each patch token
once, at the input (`pos_embed="learn"` in `vit_original.py`). That vector must then survive every
residual block (LayerNorm → attention → MLP, repeated `depth` times) as a static bias in order to
still influence attention *scores* deep in the network — nothing guarantees it does. It also has no
explicit spatial structure: the model must learn from scratch, purely via gradients on a 65×192
lookup table, that patch 9 and patch 10 are horizontally adjacent while patch 9 and patch 17 are
vertically adjacent.

**Rotary Position Embedding (RoPE)** (Su et al., *RoFormer*, 2021; adapted to vision in e.g. EVA-02,
Fang et al. 2023) takes a different approach: instead of adding a position vector to the token, it
**rotates** the query and key vectors inside *every* attention layer by an angle proportional to
position. This makes the attention score between token *i* and token *j* a function of their
**relative** position `(row_i - row_j, col_i - col_j)` by mathematical construction, not something
the model has to discover from a lookup table — verified numerically in this submission (see "Known
pitfalls" above). I extended it to two dimensions with the standard *axial* factorization: half the
head channels rotate by row position, the other half by column position.

**What I expected this to do differently, concretely, before running anything:**

1. **Faster early convergence.** Relative-position structure is built into the geometry of Q·K
   rather than something the model must discover via gradients on a lookup table, so I expected
   lower training loss in the first several epochs relative to Original.
2. **Comparable or better final test accuracy**, since CIFAR-10's natural photographs have
   meaningful local/translational structure (small objects, small spatial shifts don't change the
   object class) that a relative-position bias is well suited to exploit.
3. **More stable early training**, since RoPE reinforces position at *every* layer rather than
   relying on a single input-level signal surviving `depth` residual connections.
4. **A candidate downside:** RoPE has zero learnable positional parameters, so if the task benefited
   from a non-uniform, data-driven notion of position — which a large-enough learned table could in
   principle fit — RoPE's fixed geometric prior could be a mild disadvantage relative to Original.

---

## Justification: why PhaseGrid? *(additional variant, not required)*

Both Original and RoPE let position influence **attention scores** — i.e. *which* patches a token
looks at (Original indirectly, by biasing the token content that produces Q/K; RoPE directly, by
rotating Q/K). PhaseGrid tests a different hypothesis: what if position instead shaped **what gets
aggregated once attention has already decided where to look**, leaving the attention weights
themselves purely content-driven?

Concretely: precompute a fixed 2D-DFT positional "fingerprint" per grid position (conceptually
related to the classic sinusoidal embeddings in the original Transformer paper, but genuinely 2D and
used differently here), and feed it through a small **learned FiLM head** (Perez et al., *FiLM*,
2018) that produces a per-position gate applied multiplicatively and additively to attention's
**V** — not to the input tokens, and not to Q/K.

**What I expected this to do, concretely, before running anything:**

1. **Slower to start using position than RoPE**, because of the deliberately gentle FiLM
   initialization (`PhaseGridFiLM.__init__`, `vit_phasegrid.py` L101–108: zero-bias, small weight
   std, so `gamma≈1, beta≈0` at the start of training) — the model begins close to *not* using
   position at all and has to earn its way into using it.
2. **A genuinely uncertain final outcome relative to RoPE.** My best guess going in was that this
   *underperforms* RoPE, because not influencing attention scores at all means the model can't use
   position to decide *where to look* — arguably the more important role position plays in vision.
   I was not confident of this; it's plausible that decoupling "where to look" (content-only) from
   "what you get" (position-aware) produces cleaner representations that generalize better.
3. **More learned capacity than RoPE (58,752 vs. 0 params)** but a live risk of that capacity
   showing up as overfitting rather than better fit, on a dataset this size trained from scratch.

---

## Discussion: did results match expectations?

*(Fill in after training completes on the real CIFAR-10 data — replace bracketed values
with your actual numbers from `compare_results.py`. Keep this honest: report what actually happened
even where it didn't match the prediction above.)*

**Final results:** Original = **[X.XX]%** top-1 test accuracy, Modified (RoPE) = **[Y.YY]%**.
Delta = **[+/-Z.ZZ] pp**.

**Expectation 1 (faster early convergence):** [Compare `train_loss` at epoch 5 and epoch 10 between
the two `results_*.json` histories — state whether RoPE was ahead, and whether the gap persisted or
closed by the final epoch.]

**Expectation 2 (comparable-or-better accuracy):** [State the direction and size of the final gap.
A gap under roughly 1 percentage point at this model/dataset scale should be treated as within
run-to-run variance for a single seed rather than a decisive result — re-running with 2-3 seeds per
variant would be needed to say more confidently which is better.]

**Expectation 3 (more stable early training):** [Look at the loss plot — was RoPE's validation loss
curve visibly smoother/less noisy in the first 10-20 epochs than Original's?]

**Expectation 4 (RoPE's fixed-prior downside):** [State whether Original ever pulled ahead of RoPE
at any point in training, which would support this concern; if not, note that a fixed geometric
prior appears sufficient at this data scale.]

**Overall assessment (Original vs. RoPE):** [2-4 sentences: which predictions held, which didn't,
and the most likely explanation for any mismatch — e.g., dataset/resolution may be too small to
strongly differentiate the two schemes; RoPE's advantages are typically more pronounced at longer
sequence lengths / higher resolution than CIFAR-10's 32×32, 8×8-patch grid provides.]

### PhaseGrid vs. both baselines *(additional, not required)*

**Expectation 1 (slower to start using position):** [Compare PhaseGrid's early-epoch `train_loss`/
`val_loss` against Original and RoPE — does it track close to Original initially (position barely
used) before diverging, consistent with the gentle FiLM init?]

**Expectation 2 (uncertain vs. RoPE, my prior guess was PhaseGrid underperforms):** Final:
PhaseGrid = **[Z.ZZ]%**. [State whether PhaseGrid beat, matched, or lost to RoPE and to Original —
and whether that matches the stated prior guess.]

**Expectation 3 (extra capacity → possible overfitting):** [Compare final train/val loss gap for
PhaseGrid against the other two variants — a much larger gap would suggest the extra FiLM capacity
(58,752 params) is overfitting rather than helping.]
