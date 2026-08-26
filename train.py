"""
train.py

Shared training script for ALL THREE ViT variants. Which model gets built
is controlled entirely by the --variant flag ('original', 'modified', or
'phasegrid'); every other hyperparameter (epochs, LR schedule, batch
size, augmentation, weight decay, label smoothing, seed) is identical
across all three calls, so any accuracy/loss difference between runs of
this script is attributable to the positional embedding mechanism and
nothing else.

Usage:
    python train.py --variant original  --epochs 60
    python train.py --variant modified  --epochs 60   # RoPE
    python train.py --variant phasegrid --epochs 60   # novel FiLM-on-V scheme

Dataset: CIFAR-10 (torchvision.datasets.CIFAR10) -- 10 classes, 32x32 RGB
images, 50,000 train / 10,000 test images. CIFAR-10 is one of the
benchmark datasets explicitly reported in Dosovitskiy et al. 2021 (Table
2 / Table 5), unlike Tiny-ImageNet-200 which is NOT mentioned anywhere in
the paper -- this satisfies the assignment's "choose any one benchmark
dataset reported in the paper" requirement literally. Auto-downloads via
torchvision (download=True), no manual license/download step. The last
5,000 training images are held out as a validation split (CIFAR-10 ships
with only train/test, no official validation split) for checkpoint
selection; final accuracy is reported on the untouched 10,000-image test
set. Trained from scratch, no pretrained weights.
"""

import argparse
import copy
import math
import time
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from vit_original import build_vit_original, count_params as count_params_original
from vit_modified import build_vit_modified, count_params as count_params_modified
from vit_phasegrid import build_vit_phasegrid, count_params as count_params_phasegrid


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
VAL_SIZE = 5000  # held out from the 50,000-image train split


def build_test_loader(img_size: int, batch_size: int = 512):
    """Builds only the CIFAR-10 test-set DataLoader, with no train/val
    logic involved -- used by make_figures.py so it doesn't need to
    reconstruct (or accidentally depend on the size of) the train/val
    split just to get test-set images for the confidence histogram."""
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=eval_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return test_loader


def build_dataloaders(img_size: int, batch_size: int, seed: int):
    train_transform = T.Compose([
        T.RandomCrop(img_size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    print("Loading CIFAR-10 (torchvision.datasets.CIFAR10)...")
    full_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform)
    full_train_eval = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=eval_transform)
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=eval_transform)

    # CIFAR-10 ships with only train/test (no official validation split).
    # Hold out the last VAL_SIZE training images (fixed by seed via the
    # dataset's own deterministic ordering, not shuffled here) for
    # checkpoint selection; train on the rest.
    train_idx = list(range(0, len(full_train) - VAL_SIZE))
    val_idx = list(range(len(full_train) - VAL_SIZE, len(full_train)))

    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_train_eval, val_idx)  # eval transform, no augmentation

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = F.cross_entropy(out, y, reduction="sum")
        total_loss += loss.item()
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        args.img_size, args.batch_size, args.seed)

    torch.manual_seed(args.seed)  # re-seed immediately before model init
    if args.variant == "original":
        model = build_vit_original(
            img_size=args.img_size, patch_size=args.patch_size,
            num_classes=NUM_CLASSES, embed_dim=args.embed_dim, depth=args.depth,
            num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
        ).to(device)
        count_fn = count_params_original
    elif args.variant == "modified":
        model = build_vit_modified(
            img_size=args.img_size, patch_size=args.patch_size,
            num_classes=NUM_CLASSES, embed_dim=args.embed_dim, depth=args.depth,
            num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
        ).to(device)
        count_fn = count_params_modified
    elif args.variant == "phasegrid":
        model = build_vit_phasegrid(
            img_size=args.img_size, patch_size=args.patch_size,
            num_classes=NUM_CLASSES, embed_dim=args.embed_dim, depth=args.depth,
            num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
        ).to(device)
        count_fn = count_params_phasegrid
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    pos_params, total_params = count_fn(model)
    print(f"[{args.variant}] total params: {total_params:,}  "
          f"positional-specific params: {pos_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        running_loss, n_seen = 0.0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=args.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_loss = running_loss / n_seen
        val_loss, val_acc = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # >= (not >) so the FIRST epoch always sets best_state, even if
        # val_acc is 0.0 -- guarantees best_state is never left as None,
        # regardless of how validation accuracy behaves early in training.
        if val_acc >= best_val_acc or best_state is None:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        dt = time.time() - t0
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[{args.variant:10s}] epoch {epoch+1:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc*100:.2f}%  ({dt:.1f}s)")

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[{args.variant}] FINAL test_loss={test_loss:.4f}  "
          f"test_acc={test_acc*100:.2f}%")

    # Save history + final metrics + checkpoint for later plotting/comparison.
    out = {
        "variant": args.variant,
        "history": history,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_params": total_params,
        "positional_params": pos_params,
        "args": vars(args),
    }
    with open(f"results_{args.variant}.json", "w") as f:
        json.dump(out, f, indent=2)
    torch.save(model.state_dict(), f"checkpoint_{args.variant}.pt")
    print(f"Saved results_{args.variant}.json and checkpoint_{args.variant}.pt")


def build_argparser():
    p = argparse.ArgumentParser(description="Train original or modified ViT on CIFAR-10.")
    p.add_argument("--variant", type=str, required=True, choices=["original", "modified", "phasegrid"])
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=9)
    p.add_argument("--num_heads", type=int, default=3)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
