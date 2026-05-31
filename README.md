# waste-cv-classifier

Intelligent vision-based decision support for automated residential waste classification.

[![Open Baseline Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ishank-dubey/waste-cv-classifier/blob/main/notebooks/01_baseline_waste_classifier.ipynb)

## Purpose

This repository supports PhD research on computer vision for residential waste classification. The first goal is to build a clean baseline image-classification pipeline, then extend it with reliability-focused experiments.

Patent-sensitive implementation details should stay out of public commits until the filing strategy is clear.

## Repository Structure

```text
waste-cv-classifier/
├── notebooks/              # Colab notebooks
├── src/                    # Reusable Python code
├── data/                   # Local dataset folders, ignored by Git
├── outputs/                # Local experiment outputs, ignored by Git
└── docs/                   # Research notes and non-sensitive documentation
```

## Colab Workflow

Open the baseline notebook directly:

https://colab.research.google.com/github/Ishank-dubey/waste-cv-classifier/blob/main/notebooks/01_baseline_waste_classifier.ipynb

Inside Colab, use:

```bash
!git clone https://github.com/Ishank-dubey/waste-cv-classifier.git
%cd waste-cv-classifier
```

If the repo is already cloned in the Colab session:

```bash
%cd waste-cv-classifier
!git pull
```

## First Milestone

Train a baseline classifier for common waste categories and record:

- dataset name and class labels,
- model architecture,
- training accuracy and validation accuracy,
- precision, recall, and F1-score,
- confusion matrix,
- examples of wrong classifications.
