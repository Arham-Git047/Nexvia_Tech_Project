"""Train a small 3D MONAI U-Net on a Medical Segmentation Decathlon-style NIfTI folder.

Expected files: data/raw/msd/imagesTr/*.nii[.gz] and labelsTr/*.nii[.gz].
This is a research training entry point and has not been run against a downloaded dataset here.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, RandFlipd, RandRotate90d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/raw/msd"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--roi", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--output", type=Path, default=Path("models/monai_unet_msd.pth"))
    p.add_argument("--classes", type=int, default=2, help="Background plus target classes for the selected MSD task")
    args = p.parse_args()
    image_dir, label_dir = args.data_root / "imagesTr", args.data_root / "labelsTr"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(f"Expected NIfTI imagesTr and labelsTr under {args.data_root}. Read data/README.md first.")
    images = sorted(list(image_dir.glob("*.nii.gz")) + list(image_dir.glob("*.nii")))
    items = []
    for image in images:
        candidates = [label_dir / image.name, label_dir / image.name.replace(".nii.gz", ".nii")]
        label = next((x for x in candidates if x.is_file()), None)
        if label: items.append({"image": str(image), "label": str(label)})
    if len(items) < 2: raise SystemExit("Fewer than two paired image/label volumes found; check the dataset task layout")
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
    ])
    split = max(1, int(len(items) * 0.8))
    train_ds = CacheDataset(data=items[:split], transform=transforms, cache_rate=0.1, num_workers=2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(spatial_dims=3, in_channels=1, out_channels=args.classes, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in train_loader:
            x = batch["image"].to(device).float(); y = batch["label"].to(device).long()
            optimizer.zero_grad(set_to_none=True); logits = model(x); loss = loss_fn(logits, y); loss.backward(); optimizer.step()
            losses.append(float(loss.item()))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "device": str(device)}
        history.append(record); print(json.dumps(record))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_version": "monai-3d-unet-msd-research-v1", "classes": args.classes, "roi": args.roi}, args.output)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/segmentation_training_history.json").write_text(json.dumps({"history": history, "rows_train": split, "rows_available": len(items), "warning": "Train loss alone is not a segmentation performance metric; evaluate Dice/HD95 on a held-out patient-level test split before making claims."}, indent=2), encoding="utf-8")
    print(f"Saved research checkpoint: {args.output}")

if __name__ == "__main__": main()
