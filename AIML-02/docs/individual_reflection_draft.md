# Individual Reflection — Draft to Personalize

**Student:** Arham Kelkar  
**Institution:** Dr. D.Y. Patil School of Science and Technology  
**Project:** AIML-02 — AI-Assisted Medical Imaging and Diagnosis System

> Personalize this draft to match your own experience and responsibilities before submission.

## What I worked on

I assembled and ran the MedVision Review local research prototype with a Dockerized FastAPI backend and React frontend. I completed a rapid DenseNet121 multi-label classifier experiment using a limited ChestMNIST sample, saved the model checkpoint and metrics, and connected the checkpoint to the API. I tested a held-out sample through the model service and browser interface, fixed a Matplotlib Grad-CAM compatibility issue, and verified that a heatmap activation overlay was generated. I also exercised the human review workflow by recording a reviewer action and note.

## What I learned

I learned how to connect a model-training pipeline to a serving API, package the application with Docker, manage model checkpoints, and generate Grad-CAM activation overlays. I also learned that a successful UI workflow is not the same as clinical validation: small data subsets, low-resolution images, uncalibrated scores and potential dataset bias limit what can be claimed.

## Challenges encountered

The API initially failed during Grad-CAM overlay generation because the installed Matplotlib version did not provide `matplotlib.cm.get_cmap`. I changed the colormap call to a compatible `cm.jet` API, copied the updated module into the running container and verified that prediction and heatmap generation succeeded. The project also required careful control of training scope because the available computer had limited resources and time.

## Remaining work and limitations

The quick classifier uses 800/200/200 samples from ChestMNIST and only two epochs. The separate MONAI 3D U-Net segmentation path has not been trained; Orthanc DICOM ingestion has not been verified end-to-end. There is no external or clinical validation. These limits must remain clear in the README and presentation.

## Before submitting

- Add the exact tasks you personally completed and what you owned.
- Add the actual screenshots/video you recorded yourself.
- Do not claim MONAI segmentation or Orthanc validation unless you complete and test them.
- Do not present model outputs as diagnoses or calibrated clinical probabilities.
