# ViT Positional Embedding Comparison on CIFAR-10

This project compares the original ViT positional embedding with 2D axial RoPE. All models use
timm's `VisionTransformer` (`timm==1.0.28`) and are trained from scratch on CIFAR-10. The only
thing that changes between variants is how position information enters the model.

A third variant, **PhaseGrid**, is included as an extra experiment. The required comparison is
`vit_original.py` vs. `vit_modified.py`; `vit_phasegrid.py` can be ignored if only that is being
graded.

## Files

| File | Purpose |
|---|---|
| `vit_original.py` | Original ViT with timm's learned absolute positional embedding |
| `vit_modified.py` | ViT with the positional embedding replaced by 2D axial RoPE |
| `vit_phasegrid.py` | Extra variant: fixed 2D Fourier position fingerprint → learned FiLM on V |
| `train.py` | Shared training script; `--variant original\|modified\|phasegrid` |
| `compare_results.py` | Prints the accuracy table and saves the combined loss plot |
| `make_figures.py` | Generates the report figures (loss, validation accuracy, confidence histogram) |

## How to run

```bash
pip install torch torchvision timm matplotlib

python train.py --variant original --epochs 60
python train.py --variant modified --epochs 60
python train.py --variant phasegrid --epochs 60   # optional

python compare_results.py
python make_figures.py                            # figures saved to ./figures/
```

`make_figures.py` works with any number of trained variants and skips missing ones. The
confidence histogram reloads each saved checkpoint and runs one pass over the test set.

## Setup

- **Dataset:** CIFAR-10 (one of the benchmarks reported in Dosovitskiy et al., 2021). 50,000 train
  / 10,000 test images; the last 5,000 training images are held out for validation and
  checkpoint selection. Downloaded automatically by torchvision.
- **Model:** ViT with patch size 4 (8×8 patch grid), embed dim 192, depth 9, 3 heads, MLP ratio 4.
- **Training:** identical for all variants. AdamW, cosine LR schedule with warmup, 60 epochs,
  batch size 256, weight decay 0.05, label smoothing 0.1, seed 42.
- **Evaluation:** top-1 accuracy on the CIFAR-10 test set, using each run's best-validation
  checkpoint.

## What changes between the models

**Original:** timm's default, `pos_embed="learn"`. A learned 65×192 table (64 patches + CLS) is
added to the tokens once, at the input.

**Modified (RoPE):** two changes to the constructor call, everything else identical:

```python
pos_embed="none",            # turn off the learned table
attn_layer=RoPEAttention,    # use the RoPE attention layer
```

`RoPEAttention` subclasses timm's `Attention`. Before computing attention scores it rotates the
patch tokens' Q and K according to their row and column position. The CLS token is not rotated.
Half of each head's channels encode the row and the other half the column. The sin/cos tables are
computed once and attached to every attention layer, so timm's `Block` class did not need changing.

`RoPEAttention` uses timm's standard (non-fused) attention path, because RoPE has to modify Q and K
before the score matrix is computed. Both models use timm's default settings otherwise
(`qk_norm=False`, no gating), so the comparison stays fair.

**PhaseGrid (extra):** same pattern, `pos_embed="none"` and `attn_layer=PhaseGridAttention`. Each
position gets a fixed 2D Fourier fingerprint (sin/cos of its row and column at 4 frequencies). A
small learned linear layer in every block turns that fingerprint into a scale and shift applied to
**V**. Attention scores are left untouched, so position changes *what is collected* rather than
*where the model looks*.

## Parameter check

Each model file has a self-test (`python vit_original.py`, etc.) that prints these counts:

| Variant | Total params | Positional params | Backbone params |
|---|---|---|---|
| Original | 4,028,170 | 12,480 | 4,015,690 |
| RoPE | 4,015,690 | 0 | 4,015,690 |
| PhaseGrid | 4,074,442 | 58,752 | 4,015,690 |

The backbone is identical in all three. PhaseGrid's 58,752 come from one `Linear(16, 384)` per block:
(16×384 + 384) × 9 blocks.

## Bugs caught during development

1. **RoPE axis mixing.** The first version applied a single `rotate_half` across the whole head,
   which mixed row channels with column channels and broke the relative-position property. The
   fix rotates the row half and the column half separately. A numerical check confirms that two
   query/key pairs with the same relative offset (+1, +2) from different absolute positions now give
   the same attention score.
2. **PhaseGrid frequency aliasing.** On an 8-wide grid, frequencies above 4 are sign-flipped copies of
   lower ones (sin(2πfn/N) = −sin(2π(N−f)n/N)), so they add no new information. The fingerprint
   builder now asserts `num_freqs <= grid_size // 2`.

## Why RoPE?

The learned table is added once at the input and has to survive 9 residual blocks to still affect
attention deep in the network. It also has no built-in spatial structure: the model has to learn
from scratch which patches are neighbours.

RoPE (Su et al., 2021) instead rotates Q and K in every attention layer by an angle that depends on
position. After rotation, the dot product between two tokens depends only on their relative offset
(Δrow, Δcol). Relative position is built in by construction, and it costs zero parameters.

Expected before training:

1. Faster early convergence.
2. Similar or better final accuracy.
3. Smoother early training, since position is reinforced at every layer.
4. Possible downside: with no learnable position parameters, RoPE cannot learn a non-uniform notion
   of position.

For PhaseGrid, the expectation was that it would start slowly (its FiLM layers start near identity)
and probably underperform RoPE, since it cannot use position to decide where to look.

## Results

| Variant | Top-1 accuracy | Test loss | Extra params |
|---|---|---|---|
| Original | 80.22% | 0.6457 | 12,480 |
| **RoPE** | **83.25%** | **0.5848** | **0** |
| PhaseGrid | 80.31% | 0.6618 | 58,752 |

## Discussion

1. **Faster convergence: confirmed.** RoPE's loss separates from Original's within about 5 epochs
   and stays lowest. Its validation accuracy leads by roughly 5–8 points between epochs 10 and 40.
2. **Better accuracy: confirmed.** +3.03 points over Original, with lower test loss. The gap is
   visible throughout training, not just at the end.
3. **Smoother early training: partly confirmed.** RoPE's early loss curve is smoother, but the
   difference is small and disappears by about epoch 40.
4. **Fixed-prior downside: not observed.** Original never leads RoPE at any epoch, so a fixed
   geometric prior seems sufficient at this scale (32×32 images, 8×8 grid).

RoPE's predictions are also more confident: its test-set confidence histogram has a taller, narrower
peak near 0.9.

**PhaseGrid** matched Original (80.31% vs. 80.22%) and lost to RoPE, as expected. Its extra 58,752
parameters did not improve accuracy, and it had the highest test loss.

**Limitation:** all results are from a single seed. Running 2–3 seeds per variant would be needed
to report mean ± standard deviation.
