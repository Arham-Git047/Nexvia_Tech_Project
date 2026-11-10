# Test Results and Validation Report — MedVision Review

**Date:** 10 November 2026  
**Developer:** Arham Kelkar  
**Institution:** Dr. D.Y. Patil School of Science and Technology

> This report separates software/workflow verification from model-performance claims. No clinical validation is claimed.

## 1. Summary

The local Docker build and core classification/review demonstration succeeded on the target Windows computer. The DenseNet121 model trained on a small ChestMNIST subset, loaded in the API, produced an inference output and a Grad-CAM overlay, and the React interface displayed the result. The reviewer action was completed. Orthanc end-to-end ingestion and MONAI segmentation remain unverified.

## 2. Environment and container checks

| Check | Observed result |
|---|---|
| Docker image build | API and frontend images built |
| Compose services | API container healthy; frontend running |
| API health | `status=ok` |
| Model status | `available=true` |
| Model version | `densenet121-chestmnist-rapid-research-v1` |
| Explainability status | `true` |
| Clinical-use flag | `false` |
| UI | Loaded at `http://localhost:8080` |
| API docs | Available at `http://localhost:8000/docs` |

## 3. Training run

- Dataset: MedMNIST ChestMNIST, 14 labels; official splits sampled deterministically.
- Train/validation/test samples: 800/200/200.
- Architecture: ImageNet-pretrained DenseNet121; frozen feature extractor; classifier head trained.
- Input: 128×128; weighted BCEWithLogitsLoss; AdamW.
- Device: CPU; epochs: 2; reported training time: 2.637 minutes (data and weight downloads additional).

| Metric | Value |
|---|---:|
| Macro average precision | 0.1185728 |
| Macro ROC-AUC | 0.6023526 |
| Micro precision @ 0.5 | 0.0872939 |
| Micro recall @ 0.5 | 0.6293706 |
| Micro F1 @ 0.5 | 0.1533220 |

These are preliminary subset metrics. They are not full NIH benchmark results or clinical validation.

## 4. Inference and explanation smoke test

- Model loaded: true.
- Top-scoring finding for one held-out sample: Pneumothorax.
- Raw model score: 0.686565 (not calibrated probability).
- Grad-CAM overlay generated: yes; output PNG size 1,578 bytes in direct service smoke test.
- React UI displayed findings and Grad-CAM activation overlay.
- The first inference attempt exposed a Matplotlib API compatibility issue (`cm.get_cmap` missing). The source was patched to `cm.jet`; subsequent direct inference and UI test succeeded.

## 5. Review and audit workflow

The UI created a case (observed ID `89205789`) with initial state `Awaiting Review`. Reviewer identifier `reviewer-01` and a research-only workflow note were entered; the user confirmed the review action completed successfully. The API source records case creation and case review events in JSONL and persists case status in the local registry. Reviewer ID is an audit label, not a verified identity.

## 6. Automated API tests

The original included API test suite contains six tests covering health, model-status fail-closed response, unsupported MIME type, missing-checkpoint behavior, unknown case and review-state transitions. The package build environment previously recorded `6 passed in 0.47s`. This was not a fresh Windows pytest run after the rapid model was added.

## 7. Not yet verified

- Orthanc DICOM server plus worker ingesting a de-identified DICOM study end-to-end.
- MONAI 3D U-Net training on a selected MSD task; no segmentation checkpoint, Dice, HD95 or segmentation overlays.
- Patient-level / external / clinical validation, calibration, subgroup evaluation and clinical safety review.
- Production deployment, public exposure, authentication, role-based access, TLS and secure audit retention.

## 8. Conclusion

The local research prototype demonstrates Dockerized API/frontend startup, small-subset DenseNet121 training, model loading, one inference, Grad-CAM generation, and the human-review workflow. It should be presented as an educational research workflow demo, not as a clinically validated diagnostic system.
