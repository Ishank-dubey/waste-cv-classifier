# Dataset V1

## Purpose

Dataset V1 is the first baseline dataset for residential waste image classification.

The goal is to train and evaluate a simple computer vision model that can classify common household waste into six categories.

## Class Labels

Use exactly these folder names:

- `organic`
- `plastic`
- `paper_cardboard`
- `metal`
- `glass`
- `other`

## Folder Structure

The project uses `torchvision.datasets.ImageFolder`, so each class must be a subfolder.

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

## Suggested Split

Start with:

- 70% training images,
- 15% validation images,
- 15% test images.

Keep the test set separate. Do not use test images while choosing model settings.

## Image Guidelines

Good baseline images should:

- show one main waste item,
- have a visible object boundary,
- avoid extreme blur,
- use realistic residential waste conditions,
- include variation in lighting and background.

## Notes to Record

For every dataset version, record:

- dataset source,
- date collected or downloaded,
- number of images per class,
- train/validation/test split,
- known class imbalance,
- known quality issues,
- preprocessing decisions.

## Git Policy

Do not commit image files to GitHub. Keep datasets local, in Google Drive, or in a separate dataset storage location.
