"""Training loops for segmentation and downstream classification tasks."""

from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import numpy as np

from ..models.unet import build_backbone
from ..data import build_dataset
from ..utils.metrics import SegmentationLoss, dice_score


# ──────────────────────────────────────────────────────────────────
#  Segmentation trainer (lung nodule)
# ──────────────────────────────────────────────────────────────────

def train_segmentation(cfg: dict[str, Any]) -> nn.Module:
    """Train a segmentation U-Net from a parsed YAML config.

    Returns the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["training"]
    mcfg = cfg["model"]

    # Model
    net = build_backbone(
        mcfg["architecture"],
        in_ch=mcfg.get("in_ch", 1),
        n_classes=mcfg.get("n_classes", 1),
    ).to(device)

    # Data
    train_ds = build_dataset(cfg, split="train")
    val_ds = build_dataset(cfg, split="val")
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"])

    # Loss & optimiser
    criterion = SegmentationLoss()
    opt = torch.optim.Adam(net.parameters(), lr=tcfg["lr"])

    # Checkpoint
    ckpt_dir = pathlib.Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, tcfg["epochs"] + 1):
        # ── Train ──────────────────────────────────────────
        net.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"seg train {ep:03d}", leave=False):
            x, y = x.to(device), y.to(device)
            if x.ndim > 4:
                x = x.squeeze(1)
            loss = criterion(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)

        train_loss = running / max(len(train_ds), 1)

        # ── Validate ───────────────────────────────────────
        net.eval()
        val_running = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if x.ndim > 4:
                    x = x.squeeze(1)
                logits = net(x)
                val_running += criterion(logits, y).item() * x.size(0)
                val_dice += dice_score(logits, y).item() * x.size(0)

        val_loss = val_running / max(len(val_ds), 1)
        avg_dice = val_dice / max(len(val_ds), 1)

        print(f"ep {ep:03d} | train {train_loss:.4f} | val {val_loss:.4f} | Dice {avg_dice:.4f}")

        if ep % cfg["checkpoint"].get("save_every", 10) == 0:
            torch.save(net.state_dict(), ckpt_dir / f"seg_ep{ep:03d}.pth")

    # Save final
    torch.save(net.state_dict(), ckpt_dir / "seg_final.pth")
    return net


# ──────────────────────────────────────────────────────────────────
#  Segmentation test evaluation
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_segmentation(
    model: nn.Module,
    cfg: dict[str, Any],
    split: str = "test",
) -> float:
    """Evaluate a segmentation model on a held-out split.

    Returns the average Dice score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    ds = build_dataset(cfg, split=split)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"])

    total_dice = 0.0
    for x, y in tqdm(loader, desc=f"seg test ({split})", leave=False):
        x, y = x.to(device), y.to(device)
        if x.ndim > 4:
            x = x.squeeze(1)
        logits = model(x)
        total_dice += dice_score(logits, y).item() * x.size(0)

    avg_dice = total_dice / max(len(ds), 1)
    print(f"Test Dice Score ({split}): {avg_dice:.4f}")
    return avg_dice


# ──────────────────────────────────────────────────────────────────
#  Full evaluation metrics for classification
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Evaluate a classifier and return accuracy, ROC-AUC, precision, recall, F1, confusion matrix."""
    from sklearn.metrics import (
        roc_auc_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    model.eval()
    all_logits, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)

    logits = torch.cat(all_logits)
    probs = logits.softmax(1)[:, 1].numpy()
    labels = torch.cat(all_labels).numpy()
    preds = logits.argmax(1).numpy()

    acc = float((preds == labels).mean())
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    prec, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    return dict(acc=acc, auc=auc, prec=float(prec), recall=float(recall), f1=float(f1), cm=cm)


# ──────────────────────────────────────────────────────────────────
#  Downstream binary classification
# ──────────────────────────────────────────────────────────────────

def train_downstream_classifier(cfg: dict[str, Any]) -> nn.Module:
    """Train a tiny CNN classifier on reconstructed slices.

    Returns the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["training"]
    mcfg = cfg.get("model", {})

    # Model
    model = build_backbone(
        mcfg.get("architecture", "tiny_cnn"),
        in_ch=mcfg.get("in_ch", 1),
        num_classes=mcfg.get("num_classes", 2),
    ).to(device)

    # Data
    train_ds = build_dataset(cfg, split="train")
    val_ds = build_dataset(cfg, split="val")
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"])

    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    ckpt_dir = pathlib.Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, tcfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        for x, y in tqdm(train_loader, desc=f"cls {ep:03d}", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()

        train_acc = correct / max(len(train_ds), 1)
        train_loss = total_loss / max(len(train_ds), 1)

        # Validate (full metrics)
        val_metrics = eval_epoch(model, val_loader, device=device)

        print(
            f"ep {ep:03d} | loss {train_loss:.4f} | acc {train_acc:.3f} | "
            f"val-acc {val_metrics['acc']:.3f}  "
            f"auc {val_metrics['auc']:.3f}  "
            f"F1 {val_metrics['f1']:.3f}"
        )

        if ep % cfg["checkpoint"].get("save_every", 5) == 0:
            torch.save(model.state_dict(), ckpt_dir / f"cls_ep{ep:03d}.pth")

    torch.save(model.state_dict(), ckpt_dir / "cls_final.pth")
    return model
