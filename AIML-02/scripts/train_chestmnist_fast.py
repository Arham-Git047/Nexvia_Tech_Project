"""Fast, research-only DenseNet121 transfer-learning run on ChestMNIST.

Uses MedMNIST ChestMNIST (NIH ChestX-ray14-derived, 28x28, 14-label).
The dataset keeps its official train/validation/test splits. Only small deterministic
subsets are used. DenseNet121 ImageNet features are frozen; the 14-output classifier
head is trained. This is a fast prototype run, not a clinical or benchmark model.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

DATA_URL = "https://zenodo.org/records/10519652/files/chestmnist.npz?download=1"
LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def choose_indices(n: int, limit: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if limit <= 0 or limit >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=limit, replace=False).astype(np.int64))


class ChestMNISTSubset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray, image_size: int):
        self.images = images
        self.labels = labels.astype(np.float32)
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        image = Image.fromarray(self.images[idx].astype(np.uint8), mode="L").convert("RGB")
        x = self.transform(image)
        y = torch.from_numpy(self.labels[idx])
        return x, y


def average_precision_binary(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_y = y[order].astype(np.float64)
    precision = np.cumsum(sorted_y) / np.arange(1, len(sorted_y) + 1)
    return float((precision * sorted_y).sum() / positives)


def auc_binary(y: np.ndarray, score: np.ndarray) -> float | None:
    y = y.astype(np.int64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate(y_true: np.ndarray, probs: np.ndarray) -> dict:
    per_label = {}
    aps, aucs = [], []
    tp = fp = fn = 0
    for i, name in enumerate(LABELS):
        y = y_true[:, i].astype(np.int64)
        p = probs[:, i]
        ap = average_precision_binary(y, p)
        auc = auc_binary(y, p)
        pred = (p >= 0.5).astype(np.int64)
        tpi = int(((pred == 1) & (y == 1)).sum())
        fpi = int(((pred == 1) & (y == 0)).sum())
        fni = int(((pred == 0) & (y == 1)).sum())
        tp += tpi; fp += fpi; fn += fni
        per_label[name] = {
            "average_precision": ap,
            "roc_auc": auc,
            "positive_support": int(y.sum()),
            "negative_support": int((y == 0).sum()),
        }
        if ap is not None: aps.append(ap)
        if auc is not None: aucs.append(auc)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "macro_average_precision": float(np.mean(aps)) if aps else None,
        "macro_roc_auc": float(np.mean(aucs)) if aucs else None,
        "micro_precision_at_0_5": float(precision),
        "micro_recall_at_0_5": float(recall),
        "micro_f1_at_0_5": float(f1),
        "labels_with_defined_ap": len(aps),
        "labels_with_defined_auc": len(aucs),
        "per_label": per_label,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("/data/chestmnist.npz"))
    p.add_argument("--output", type=Path, default=Path("/data/densenet121_nih.pth"))
    p.add_argument("--metrics", type=Path, default=Path("/data/classification_metrics.json"))
    p.add_argument("--train-limit", type=int, default=800)
    p.add_argument("--val-limit", type=int, default=200)
    p.add_argument("--test-limit", type=int, default=200)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-minutes", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    args.data.parent.mkdir(parents=True, exist_ok=True)
    if not args.data.exists():
        print(f"Downloading official ChestMNIST archive to {args.data} ...", flush=True)
        urllib.request.urlretrieve(DATA_URL, args.data)
    archive = np.load(args.data)
    required = {"train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"}
    if not required.issubset(set(archive.files)):
        raise SystemExit(f"Dataset archive missing arrays: {sorted(required - set(archive.files))}")

    train_i = choose_indices(len(archive["train_images"]), args.train_limit, args.seed)
    val_i = choose_indices(len(archive["val_images"]), args.val_limit, args.seed + 1)
    test_i = choose_indices(len(archive["test_images"]), args.test_limit, args.seed + 2)
    train_ds = ChestMNISTSubset(archive["train_images"], archive["train_labels"], train_i, args.image_size)
    val_ds = ChestMNISTSubset(archive["val_images"], archive["val_labels"], val_i, args.image_size)
    test_ds = ChestMNISTSubset(archive["test_images"], archive["test_labels"], test_i, args.image_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    weights = None if args.no_pretrained else models.DenseNet121_Weights.DEFAULT
    print(f"Loading DenseNet121 on {device}; pretrained={weights is not None}", flush=True)
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    for param in model.features.parameters():
        param.requires_grad = False
    model.to(device)

    raw_train_y = archive["train_labels"][train_i].astype(np.float32)
    positive = np.maximum(raw_train_y.sum(axis=0), 1.0)
    negative = np.maximum(len(raw_train_y) - positive, 1.0)
    pos_weight = torch.tensor(np.clip(negative / positive, 1.0, 20.0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    started = time.time()
    best_val_loss = float("inf")
    best_state = None
    epochs_completed = 0
    stopped_for_time = False
    training_steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.features.eval()  # keep frozen ImageNet feature extractor in inference mode
        losses = []
        for xb, yb in train_loader:
            if (time.time() - started) >= args.max_minutes * 60 and losses:
                stopped_for_time = True
                print("Time budget reached; finishing validation for the current partial epoch.", flush=True)
                break
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            losses.append(float(loss.item()))
            training_steps += 1
        if not losses:
            raise SystemExit("Time budget ended before any training batch completed")
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(float(criterion(model(xb.to(device)), yb.to(device)).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        epochs_completed = epoch
        print(json.dumps({"epoch": epoch, "epochs_requested": args.epochs,
                           "train_loss": float(np.mean(losses)), "val_loss": val_loss,
                           "elapsed_minutes": round((time.time() - started) / 60, 2),
                           "device": str(device)}), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stopped_for_time or (time.time() - started) >= args.max_minutes * 60:
            print("Time budget reached; stopping after validation.", flush=True)
            break

    if best_state is None:
        raise SystemExit("No checkpoint produced; training did not complete an epoch")
    model.load_state_dict(best_state)
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
            all_targets.append(yb.numpy())
    y_true = np.concatenate(all_targets, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    metrics = evaluate(y_true, probs)
    elapsed = round((time.time() - started) / 60, 3)
    metrics.update({
        "dataset": "MedMNIST ChestMNIST (NIH ChestX-ray14-derived; 28x28 educational benchmark)",
        "dataset_license": "CC BY 4.0; cite MedMNIST and its source dataset",
        "dataset_split": "official train / validation / test splits; deterministic subset from each split",
        "train_samples": len(train_ds), "validation_samples": len(val_ds), "test_samples": len(test_ds),
        "epochs_requested": args.epochs, "epochs_completed": epochs_completed,
        "training_steps": training_steps, "stopped_for_time_budget": stopped_for_time,
        "image_size": args.image_size, "architecture": "DenseNet121, ImageNet pretrained",
        "training_strategy": "frozen DenseNet feature extractor; train 14-label classifier head only",
        "loss": "weighted multi-label BCEWithLogitsLoss", "optimizer": "AdamW",
        "device": str(device), "elapsed_training_minutes": elapsed,
        "seed": args.seed, "checkpoint_path": str(args.output),
        "warning": "Rapid, low-resolution subset experiment for education only. Metrics are subset metrics, not a clinical benchmark or validation of clinical use.",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "labels": LABELS,
                "model_version": "densenet121-chestmnist-rapid-research-v1",
                "input_size": args.image_size, "seed": args.seed,
                "training_strategy": "frozen backbone; classifier head only"}, args.output)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Export a real held-out sample for upload/UI testing (not a prediction result).
    sample = Image.fromarray(archive["test_images"][int(test_i[0])].astype(np.uint8), mode="L")
    sample.save(args.data.parent / "sample_chestmnist_test.png")
    print(json.dumps({k: metrics[k] for k in ["macro_average_precision", "macro_roc_auc",
          "micro_precision_at_0_5", "micro_recall_at_0_5", "micro_f1_at_0_5",
          "train_samples", "validation_samples", "test_samples", "epochs_completed",
          "elapsed_training_minutes", "checkpoint_path"]}, indent=2), flush=True)
    print(f"Saved checkpoint: {args.output}", flush=True)
    print(f"Saved metrics: {args.metrics}", flush=True)


if __name__ == "__main__":
    main()
