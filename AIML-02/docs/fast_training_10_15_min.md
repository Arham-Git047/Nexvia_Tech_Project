# Rapid DenseNet121 training — 10–15 minute training-loop target

This run uses MedMNIST ChestMNIST (14-label, low-resolution educational dataset derived from NIH ChestX-ray14). The goal is to demonstrate the full model-loading and Grad-CAM pipeline on limited hardware, not to establish clinical performance.

## Actual configuration and result

- Train/validation/test: 800 / 200 / 200, sampled from official splits with seed 42.
- DenseNet121 initialized with ImageNet pretrained weights.
- Frozen feature extractor; only the 14-label classifier head trained.
- Input resolution: 128×128.
- Weighted multi-label BCEWithLogitsLoss; AdamW.
- CPU; two epochs; reported training time: 2.637 minutes, excluding downloads.
- Macro AP: 0.1185728; macro ROC-AUC: 0.6023526; micro P/R/F1 at 0.5: 0.0872939 / 0.6293706 / 0.1533220.

## Reproduce from the project directory (PowerShell)

```powershell
# API container must be running first.
New-Item -ItemType Directory -Force .\scripts, .\models, .\reports, .\data | Out-Null
docker cp .\scripts\train_chestmnist_fast.py medvision-review-api-1:/tmp/train_chestmnist_fast.py
docker compose exec -u 0 api python /tmp/train_chestmnist_fast.py --train-limit 800 --val-limit 200 --test-limit 200 --epochs 2 --batch-size 16 --image-size 128 --max-minutes 10

docker cp medvision-review-api-1:/data/densenet121_nih.pth .\models\densenet121_nih.pth
docker cp medvision-review-api-1:/data/classification_metrics.json .\reports\classification_metrics.json
docker cp medvision-review-api-1:/data/sample_chestmnist_test.png .\data\sample_chestmnist_test.png
docker compose restart api
```

Then check `GET /health` and `GET /model-status`. The compose file mounts `./models` read-only at `/models`.

## Limitations

This is a rapid, low-resolution subset experiment; dataset download and pretrained-weight download take additional time. The current score output is not calibrated. Do not claim full NIH training, clinical validity, or 3D segmentation results. The separate MONAI 3D U-Net training entry point requires a selected Medical Segmentation Decathlon task, paired NIfTI images and masks, task-specific preprocessing, and separate evaluation.
