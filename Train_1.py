"""
train.py

Training loop for AMCTransformer (Stage 1: train from scratch on
RadioML2016.10a). Also provides freeze/fine-tune helpers reused for
Stage 2 (2018.01A) and Stage 3 (USRP captures).

Usage (in Colab, after mounting Drive and prepping data):

    from data_utils import load_prepped
    from model import AMCTransformer
    from train import train_model

    splits, class_names = load_prepped('/content/drive/MyDrive/.../radioml2016_prepped.npz')
    model, history = train_model(splits, num_classes=len(class_names), seq_len=128)
"""

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import AMCTransformer


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class AMCDataset(Dataset):
    """Wraps prepped numpy arrays (X: (N,2,L) float, y: (N,) int) as a
    PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(splits: Dict[str, np.ndarray], batch_size: int = 256, num_workers: int = 2):
    train_ds = AMCDataset(splits["X_train"], splits["y_train"])
    val_ds = AMCDataset(splits["X_val"], splits["y_val"])
    test_ds = AMCDataset(splits["X_test"], splits["y_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    """Runs one epoch. If optimizer is provided, trains; otherwise evaluates
    (no gradient updates). Returns (avg_loss, accuracy)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

    return total_loss / total, correct / total


def train_model(
    splits: Dict[str, np.ndarray],
    num_classes: int,
    seq_len: int = 128,
    patch_size: int = 4,
    batch_size: int = 256,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 3,
    checkpoint_path: Optional[str] = "amc_transformer_stage1.pt",
    device: Optional[str] = None,
    model: Optional[nn.Module] = None,
):
    """Full Stage-1-style training run: builds (or reuses) a model, trains
    from the given data splits, and saves the best checkpoint by
    validation accuracy.

    Pass an existing `model` (e.g. one loaded from a Stage-1 checkpoint
    with encoder frozen) to reuse this same loop for Stage 2/3
    fine-tuning -- just pass a much lower `lr` and fewer `epochs`.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(splits, batch_size=batch_size)

    if model is None:
        model = AMCTransformer(seq_len=seq_len, num_classes=num_classes, patch_size=patch_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    # Linear warmup then cosine decay -- standard for transformers, helps
    # avoid early training instability from large gradients on random init.
    warmup_steps = warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup_steps:
            return (epoch + 1) / warmup_steps
        progress = (epoch - warmup_steps) / max(1, epochs - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc and checkpoint_path:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, checkpoint_path)
            print(f"  -> new best val_acc {val_acc:.4f}, checkpoint saved to {checkpoint_path}")

    print(f"\nTraining done. Best val_acc: {best_val_acc:.4f}")

    # Final test-set evaluation using the best checkpoint
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device)
    print(f"Test accuracy (best checkpoint): {test_acc:.4f}")

    return model, history


if __name__ == "__main__":
    # Quick smoke test with random dummy data -- verifies the training
    # loop runs end-to-end without shape/device errors before you spend
    # real Colab GPU time on it.
    print("Running smoke test with dummy data...")
    N_train, N_val, N_test = 512, 128, 128
    seq_len, num_classes = 128, 7

    dummy_splits = {
        "X_train": np.random.randn(N_train, 2, seq_len).astype(np.float32),
        "y_train": np.random.randint(0, num_classes, N_train),
        "X_val": np.random.randn(N_val, 2, seq_len).astype(np.float32),
        "y_val": np.random.randint(0, num_classes, N_val),
        "X_test": np.random.randn(N_test, 2, seq_len).astype(np.float32),
        "y_test": np.random.randint(0, num_classes, N_test),
    }

    model, history = train_model(
        dummy_splits, num_classes=num_classes, seq_len=seq_len,
        epochs=2, batch_size=64, checkpoint_path="smoke_test.pt",
    )
    print("Smoke test passed.")
