"""Train a research-only 14-label DenseNet121 model on NIH ChestX-ray14.

Required input: officially downloaded image files + Data_Entry_2017_v2020.csv.
The script does not download the dataset or ship any images/weights.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def labels_for(row) -> np.ndarray:
    values = set(str(row).split("|"))
    return np.asarray([1.0 if label in values else 0.0 for label in LABELS], dtype=np.float32)


class ChestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, training: bool):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5) if training else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / str(row["Image Index"])
        with Image.open(image_path) as im:
            image = self.transform(im.convert("RGB"))
        target = torch.tensor(labels_for(row["Finding Labels"]))
        return image, target


def metric_summary(y_true: np.ndarray, probs: np.ndarray) -> dict:
    per_label = {}
    ap_values, auc_values = [], []
    for i, label in enumerate(LABELS):
        truth = y_true[:, i]
        if len(np.unique(truth)) > 1:
            ap = float(average_precision_score(truth, probs[:, i]))
            auc = float(roc_auc_score(truth, probs[:, i]))
            ap_values.append(ap); auc_values.append(auc)
            per_label[label] = {"average_precision": ap, "roc_auc": auc}
        else:
            per_label[label] = {"average_precision": None, "roc_auc": None, "note": "only one class present in evaluation split"}
    return {"macro_average_precision": float(np.mean(ap_values)) if ap_values else None,
            "macro_roc_auc": float(np.mean(auc_values)) if auc_values else None,
            "per_label": per_label, "warning": "Research dataset metrics only; not evidence of clinical safety or external validity."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("models/densenet121_nih.pth"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/classification_metrics.json"))
    parser.add_argument("--max-samples", type=int, default=0, help="Optional small subset for smoke tests; not final evaluation")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet weights")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not args.csv.is_file(): raise SystemExit(f"CSV not found: {args.csv}")
    if not args.image_dir.is_dir(): raise SystemExit(f"Image directory not found: {args.image_dir}")
    df = pd.read_csv(args.csv)
    required = {"Image Index", "Finding Labels", "Patient ID"}
    missing = required.difference(df.columns)
    if missing: raise SystemExit(f"CSV missing columns: {sorted(missing)}")
    df = df[df["Image Index"].map(lambda x: (args.image_dir / str(x)).is_file())].copy()
    if args.max_samples > 0: df = df.sample(min(args.max_samples, len(df)), random_state=args.seed).copy()
    df = df.reset_index(drop=True)
    if len(df) < 20: raise SystemExit(f"Only {len(df)} readable images found; check the dataset paths")
    # Split by patient, not image, to reduce patient-level leakage.
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, hold_idx = next(splitter1.split(df, groups=df["Patient ID"].astype(str)))
    hold = df.iloc[hold_idx].reset_index(drop=True)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    val_rel, test_rel = next(splitter2.split(hold, groups=hold["Patient ID"].astype(str)))
    train_df, val_df, test_df = df.iloc[train_idx].copy(), hold.iloc[val_rel].copy(), hold.iloc[test_rel].copy()
    if args.max_samples > 0:
        # Small data sample is only for pipeline smoke tests; split may under-represent labels.
        print("WARNING: --max-samples set; treat all resulting metrics as smoke-test-only.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ChestDataset(train_df, args.image_dir, True)
    val_ds = ChestDataset(val_df, args.image_dir, False)
    test_ds = ChestDataset(test_df, args.image_dir, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    weights = None if args.no_pretrained else models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    model.to(device)
    # Positive class weighting mitigates prevalence imbalance; inspect per-label effects after training.
    train_targets = np.stack([labels_for(s) for s in train_df["Finding Labels"]])
    positives = np.maximum(train_targets.sum(axis=0), 1.0)
    negatives = np.maximum(len(train_targets) - positives, 1.0)
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device).clamp(max=20.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                val_losses.append(float(criterion(model(images), targets).item()))
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        print(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": mean_val, "device": str(device)}))
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None: raise SystemExit("Training failed; no best checkpoint available")
    model.load_state_dict(best_state); model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            logits = model(images.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy()); all_targets.append(targets.numpy())
    probs, truths = np.concatenate(all_probs), np.concatenate(all_targets)
    metrics = metric_summary(truths, probs)
    metrics.update({"rows_total": len(df), "rows_train": len(train_df), "rows_validation": len(val_df), "rows_test": len(test_df), "best_validation_loss": best_val, "epochs_requested": args.epochs, "seed": args.seed, "device": str(device), "split_unit": "Patient ID", "labels": LABELS, "checkpoint_path": str(args.output), "smoke_test_subset": args.max_samples > 0})
    args.output.parent.mkdir(parents=True, exist_ok=True); args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "labels": LABELS, "model_version": "densenet121-nih-research-v1", "input_size": 224, "seed": args.seed}, args.output)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ["macro_average_precision", "macro_roc_auc", "rows_train", "rows_validation", "rows_test", "checkpoint_path"]}, indent=2))

if __name__ == "__main__": main()
