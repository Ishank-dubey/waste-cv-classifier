# Codex Handoff

Last updated: 2026-06-29

## Project

Repository:

```text
https://github.com/Ishank-dubey/waste-cv-classifier
```

Local macOS path before migration:

```text
/Users/rajneeshdubey/Desktop/PhD/waste-cv-classifier
```

Research direction:

```text
Residential waste image classification using computer vision.
```

Initial ML goal:

```text
Build a baseline PyTorch image classifier for six residential waste classes using transfer learning.
```

Six classes:

```text
organic
plastic
paper_cardboard
metal
glass
other
```

## Current Repo Structure

Important files:

```text
README.md
requirements.txt
configs/baseline_resnet18.yaml
src/dataset.py
src/model.py
src/train.py
src/evaluate.py
docs/dataset_v1.md
docs/colab-sync.md
notebooks/01_baseline_waste_classifier.ipynb
notebooks/02_single_image_cnn_flow.ipynb
```

Purpose of notebooks:

```text
01_baseline_waste_classifier.ipynb
  Baseline training notebook. It currently contains notebook-native code and points to the six-class dataset structure.

02_single_image_cnn_flow.ipynb
  Teaching notebook for one-image CNN flow. It is intentionally self-contained and does not import src utilities yet.
```

Purpose of `src/`:

```text
src/dataset.py
  ImageFolder dataloaders and transforms.

src/model.py
  ResNet18 transfer-learning model builder.

src/train.py
  Training script. Saves best checkpoint.

src/evaluate.py
  Evaluation script. Saves confusion matrix.
```

Training command:

```bash
python src/train.py --config configs/baseline_resnet18.yaml
```

Evaluation command:

```bash
python src/evaluate.py --config configs/baseline_resnet18.yaml
```

## Dataset Layout

The code expects `torchvision.datasets.ImageFolder` layout:

```text
data/
  train/
    organic/
    plastic/
    paper_cardboard/
    metal/
    glass/
    other/
  val/
    organic/
    plastic/
    paper_cardboard/
    metal/
    glass/
    other/
  test/
    organic/
    plastic/
    paper_cardboard/
    metal/
    glass/
    other/
```

Git tracks `.gitkeep` placeholders for these folders. Actual images are ignored and should not be committed.

## Local Dataset State On iMac

HEIC images were added locally under `data/train` and converted into JPG copies using macOS `sips`.

Current local counts when this handoff was created:

```text
train/glass             jpg=19  heic=18
train/metal             jpg=22  heic=22
train/organic           jpg=19  heic=19
train/other             jpg=0   heic=0
train/paper_cardboard   jpg=23  heic=23
train/plastic           jpg=18  heic=18

val/*                   jpg=0   heic=0
test/*                  jpg=0   heic=0
```

Important:

```text
The dataset is not complete yet.
The `other` class needs images.
Validation and test sets need images.
```

Image policy:

```text
Do not commit JPG/HEIC image files to GitHub.
Use Google Drive, Hugging Face Datasets, or external storage for images.
```

## Google Drive / Colab Workflow

Preferred persistent dataset path in Google Drive:

```text
MyDrive/waste-cv-dataset/
  train/
  val/
  test/
```

Colab mount:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Drive dataset path inside Colab:

```python
from pathlib import Path

DATA_ROOT = Path("/content/drive/MyDrive/waste-cv-dataset")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"
```

For the one-image notebook, image search can use:

```python
image_paths = (
    sorted(TRAIN_DIR.glob("*/*.jpg")) +
    sorted(TRAIN_DIR.glob("*/*.jpeg"))
)
```

## Dataset Zip Created On iMac

A JPG-only dataset zip was created here:

```text
/Users/rajneeshdubey/Desktop/PhD/waste-cv-dataset-jpg.zip
```

Size at creation:

```text
297 MB
```

It preserves:

```text
waste-cv-dataset-jpg/train/...
waste-cv-dataset-jpg/val/...
waste-cv-dataset-jpg/test/...
```

It was created because the local Google Drive mount timed out while copying many files.

If uploaded to Google Drive, unzip in Colab with:

```python
from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/MyDrive/waste-cv-dataset-jpg.zip" -d /content/
```

## Learning State

Current learning focus:

```text
CNN fundamentals using one image.
```

Recently discussed concepts:

```text
PIL image size shows width and height, not channels.
PIL image mode shows color mode, usually RGB.
After transforms.ToTensor(), image shape is [channels, height, width].
After unsqueeze(0), CNN input shape is [batch, channels, height, width].
x[0, i] is equivalent to x[0][i].
RGB channels plotted separately can look similar because they show intensity maps.
```

Recommended next learning steps:

```text
1. Continue `02_single_image_cnn_flow.ipynb`.
2. Understand Conv2d, ReLU, MaxPool2d, Flatten, Linear.
3. Then connect the notebook idea back to ResNet18 transfer learning.
4. Later move reusable teaching models into `src/model.py` only after the concept is clear.
```

## Hardware Decision Context

The current iMac is slow and hard to upgrade.

Clarified laptop role:

```text
Local research workstation with light CUDA.
Heavy training will mostly run on Hugging Face/cloud.
```

Evaluated options:

```text
MacBook Air
  Good for writing and productivity, but no NVIDIA CUDA.

RTX 4060 / RTX 4070 laptops
  Better for local ML, but often too expensive.

RTX 4050 laptops
  Good compromise, but budget/config issues.

ASUS TUF A15 FA506NCG-HN251WS
  Ryzen 7 7445HS, RTX 3050 4GB, 75W TGP, 16GB RAM upgradeable, 1TB SSD.
  Considered reasonable for local coding, learning, dataset prep, light CUDA, and cloud-first training.
```

Current hardware verdict:

```text
For a cloud-first workflow, ASUS TUF A15 RTX 3050 4GB + 1TB SSD + upgradeable RAM is acceptable.
It should not be treated as a strong deep-learning training laptop.
```

## Migration To New Windows Laptop

Code migration:

```bash
git clone https://github.com/Ishank-dubey/waste-cv-classifier.git
cd waste-cv-classifier
```

Install basics:

```text
Git
VS Code
Python or Miniconda
NVIDIA driver
PyTorch with CUDA support
Google Drive Desktop if needed
Codex / ChatGPT tooling as desired
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Move dataset separately:

```text
Use Google Drive, the JPG zip, or an external SSD.
Do not rely on GitHub for dataset images.
```

Before leaving iMac:

```bash
git status
git push origin main
```

Also preserve:

```text
/Users/rajneeshdubey/Desktop/PhD/waste-cv-dataset-jpg.zip
local data/train images if needed
any uncommitted notes outside the repo
```

## Next Actions

1. Finish one-image CNN flow notebook in Colab.
2. Add images for `other`.
3. Create `val` and `test` splits.
4. Decide whether dataset will live in Google Drive or Hugging Face Datasets.
5. Update baseline config paths if training directly from Drive.
6. Run first baseline ResNet18 training once dataset has train/val/test images.
7. Save results: accuracy, confusion matrix, and wrong predictions.
