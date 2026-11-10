# AIML-02 GitHub upload notes

This folder is ready to add under the combined `Nexvia_Tech_Project/AIML-02/` repository path. The PowerPoint and video files are intentionally excluded; they are being prepared separately by the student.

## One local file to copy before commit

The trained DenseNet121 checkpoint was created on the student's Windows machine and is not embedded in this ZIP. Copy the existing local checkpoint into `AIML-02/models/densenet121_nih.pth` before committing this folder. From the existing `Nexvia_Tech_Project` Git working directory, run:

```powershell
New-Item -ItemType Directory -Force .\AIML-02\models | Out-Null
Copy-Item "$HOME\Documents\AIML-02\AI_Medical_Imaging_Submission\project\models\densenet121_nih.pth" .\AIML-02\models\densenet121_nih.pth -Force
```

The `.gitignore` inside AIML-02 allows this exact checkpoint filename and excludes full datasets, NPZ archives, DICOM/NIfTI files, runtime state and credentials. The recorded local checkpoint was approximately 28.4 MB, below GitHub's 50 MiB recommended warning threshold.

## Stage and publish after extraction

```powershell
git add .\AIML-02
git status --short
git diff --cached --check
git commit -m "Add AIML-02 medical imaging research prototype"
git push
```

Before pushing, confirm the staged files include the checkpoint, source, README, docs, architecture diagram and metrics. Do not include the full ChestMNIST/NIH archive or patient-derived runtime data.

## Status caveats

The verified local prototype includes Docker API/frontend startup, 800/200/200 ChestMNIST rapid training, API model loading, one inference, Grad-CAM display, and a reviewer workflow action. The MONAI 3D U-Net training and Orthanc end-to-end DICOM workflow remain unverified. The PowerPoint and recording are not in this pack.
