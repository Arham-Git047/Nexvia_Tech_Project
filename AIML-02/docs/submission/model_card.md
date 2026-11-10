# Model Card and Limitations — MedVision Review

**Developer:** Arham Kelkar  
**Institution:** Dr. D.Y. Patil School of Science and Technology  
**Model version:** `densenet121-chestmnist-rapid-research-v1`

> **Educational/research only — not for clinical use.** This card describes an experimental classifier. It is not a medical device or a diagnostic claim.

## 1. Model identity

- Architecture: TorchVision DenseNet121.
- Output: 14 sigmoid scores, aligned to the 14 ChestMNIST/NIH-style label names in the source.
- Initialization: ImageNet pretrained DenseNet121 weights.
- Training strategy: feature extractor frozen; final 14-output classifier head trained.
- Objective: weighted multi-label `BCEWithLogitsLoss`.
- Optimizer: AdamW.
- Input: ChestMNIST sample resized to 128×128 and normalized using ImageNet channel statistics.
- Explainability: Grad-CAM from the final DenseNet dense block; overlay is rendered using a jet colour map.

## 2. Dataset and sampling

Rapid run used MedMNIST ChestMNIST, a low-resolution 28×28, 14-label chest-X-ray educational benchmark derived from NIH ChestX-ray14. Deterministic subsets were sampled from the official splits: 800 train, 200 validation, 200 test (seed 42). The full dataset was not used for this rapid run. MedMNIST explicitly states that it is not intended for clinical use, in part because downsampling can remove pathology detail. MedMNIST is listed under CC BY 4.0; cite MedMNIST v2 and the source dataset when using the data.

## 3. Evaluation results

| Metric | Result |
|---|---:|
| Macro average precision | 0.1185728 |
| Macro ROC-AUC | 0.6023526 |
| Micro precision at 0.5 | 0.0872939 |
| Micro recall at 0.5 | 0.6293706 |
| Micro F1 at 0.5 | 0.1533220 |

Training completed two epochs on CPU; script-reported training time was 2.637 minutes (excluding downloads). The checkpoint selected the epoch with the best validation loss. Full recorded values are in `reports/classification_metrics.json`.

## 4. Demonstrated inference

One held-out sample generated top output `Pneumothorax` with score 0.686565. The Grad-CAM PNG was generated (1,578 bytes in a direct service test), displayed in the React UI, and a clinician-review workflow action was completed. These are workflow smoke-test observations, not estimates of diagnostic accuracy. Scores are not guaranteed to be calibrated probabilities.

## 5. Intended use

Educational exploration of multi-label image classification, model-serving, visual activation maps, reviewer queues, and audit logging using public research data.

## 6. Out-of-scope uses

Diagnosis, triage, treatment selection, screening, patient prioritization, or replacing a qualified clinician. Do not use on live patient records or use scores as the sole basis of care decisions.

## 7. Limitations and risks

- Small sample count and only two epochs; likely high uncertainty and unstable per-label metrics.
- Low-resolution source images (28×28) may not preserve clinically important features.
- Weak-label quality, class imbalance, distribution shift, scanner/site differences and subgroup gaps can affect performance.
- No external validation, patient-level audit of splits, prospective evaluation or calibration study is claimed in this rapid run.
- Grad-CAM visualizes network activation; it is not a lesion mask, causal explanation or proof of pathology.
- Review identifier is not authenticated; local JSON and JSONL storage is not a production audit infrastructure.
- Segmentation branch (MONAI 3D U-Net) has not been trained; no Dice or HD95 values are available.

## 8. Monitoring and governance required before any real-world use

A larger approved dataset, patient-level splits, label quality review, external validation, sensitivity/specificity and per-label metrics, calibration, subgroup analysis, human factors testing, secure access control, encryption, privacy controls, audit retention, regulatory and ethics assessment would be required. A disclaimer alone does not make the system safe for clinical use.

## 9. References

- MedMNIST official repository and usage notes: https://github.com/MedMNIST/MedMNIST
- Yang, J. et al. (2023), *MedMNIST v2*, Scientific Data: https://doi.org/10.1038/s41597-022-01721-8
- NIH ChestX-ray14 source: https://nihcc.app.box.com/v/ChestXray-NIHCC
