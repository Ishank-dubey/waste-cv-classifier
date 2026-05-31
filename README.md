# waste-cv-classifier

Baseline PyTorch image-classification project for residential waste classification using transfer learning.

[![Open Baseline Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ishank-dubey/waste-cv-classifier/blob/main/notebooks/01_baseline_waste_classifier.ipynb)

## Project Goal

This project builds a simple, readable baseline classifier for common residential waste categories:

- `organic`
- `plastic`
- `paper_cardboard`
- `metal`
- `glass`
- `other`

The first model uses **ResNet18 transfer learning** from `torchvision`. The training script saves the best validation checkpoint, and the evaluation script reports test accuracy plus a confusion matrix.

Patent-sensitive implementation details should stay out of public commits until the filing strategy is clear. This repository should begin with generic baseline experiments only.

## Repository Structure

```text
waste-cv-classifier/
├── configs/
│   └── baseline_resnet18.yaml
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── docs/
│   └── dataset_v1.md
├── notebooks/
├── outputs/
└── src/
    ├── dataset.py
    ├── evaluate.py
    ├── model.py
    └── train.py
```

## Dataset Structure

This project uses `torchvision.datasets.ImageFolder`, so each class must be a folder.

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

Do not commit image datasets to GitHub. Keep them local or mount them from Google Drive in Colab.

## Setup

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Colab, open the notebook:

https://colab.research.google.com/github/Ishank-dubey/waste-cv-classifier/blob/main/notebooks/01_baseline_waste_classifier.ipynb

## Training

Train the baseline ResNet18 model:

```bash
python src/train.py --config configs/baseline_resnet18.yaml
```

The script will:

- load `data/train` and `data/val`,
- use CPU or GPU automatically,
- train a ResNet18 transfer-learning classifier,
- print train and validation accuracy after each epoch,
- save the best checkpoint to `outputs/best_resnet18.pt`.

## Evaluation

Evaluate the best checkpoint on the test set:

```bash
python src/evaluate.py --config configs/baseline_resnet18.yaml
```

The script will:

- load `data/test`,
- print test accuracy,
- print a classification report,
- save a confusion matrix image to `outputs/confusion_matrix.png`.

## First Research Baseline

Record the following for the first experiment:

- dataset source,
- number of images per class,
- train/validation/test split,
- model architecture,
- number of epochs,
- best validation accuracy,
- test accuracy,
- confusion matrix,
- examples of incorrect predictions.
