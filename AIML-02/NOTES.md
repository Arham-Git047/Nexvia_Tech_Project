# GitHub Upload Notes

This AIML-02 folder is prepared for placement under the combined `Nexvia_Tech_Project/AIML-02/` directory. The PowerPoint and video files are intentionally not included; you said you will manage those separately.

Before pushing, copy your actual trained checkpoint from the local completed project to `AIML-02/models/densenet121_nih.pth`. The generated checkpoint was saved locally during your run and is not embedded in this ZIP because it was created on your Windows computer. The `.gitignore` explicitly allows this exact checkpoint filename while excluding datasets and runtime data.

Also copy `reports/classification_metrics.json` from your local run if you want the exact per-label metrics. This pack includes the observed aggregate metrics from your training console.

The README and documents distinguish verified work from incomplete items: local Docker API/frontend, rapid classifier training, inference, Grad-CAM and review workflow were verified; Orthanc end-to-end and MONAI 3D U-Net training were not.
