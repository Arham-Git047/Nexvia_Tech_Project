# Data directory

This repository does not distribute the full NIH ChestX-ray14 or MedMNIST ChestMNIST archives, patient images, or derived datasets. The fast training script downloads the official ChestMNIST archive into the container data volume (`/data/chestmnist.npz`) and exports one held-out sample image for local UI verification.

ChestMNIST is a low-resolution, 14-label educational benchmark derived from NIH ChestX-ray14. MedMNIST is explicitly not intended for clinical use. Review the CC BY 4.0 terms and cite MedMNIST v2 and the original source dataset.

Official resources: https://github.com/MedMNIST/MedMNIST and https://doi.org/10.1038/s41597-022-01721-8
