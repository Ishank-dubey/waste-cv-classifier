import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tqdm import tqdm

from dataset import create_test_loader
from model import build_model


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().numpy()

        all_predictions.extend(predictions)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_predictions)


def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Waste Classifier Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate(config):
    device = get_device()
    print(f"Using device: {device}")

    test_loader, class_names = create_test_loader(config)
    model = build_model(config).to(device)

    checkpoint_path = config["outputs"]["checkpoint_path"]
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    checkpoint_classes = checkpoint.get("class_names")
    if checkpoint_classes:
        class_names = checkpoint_classes

    y_true, y_pred = collect_predictions(model, test_loader, device)
    accuracy = (y_true == y_pred).mean()

    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    confusion_matrix_path = config["outputs"]["confusion_matrix_path"]
    save_confusion_matrix(y_true, y_pred, class_names, confusion_matrix_path)
    print(f"Saved confusion matrix to {confusion_matrix_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained waste classifier.")
    parser.add_argument(
        "--config",
        default="configs/baseline_resnet18.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(load_config(args.config))
