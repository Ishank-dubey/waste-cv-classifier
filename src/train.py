import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from dataset import create_dataloaders
from model import build_model


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one_epoch(model, dataloader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        predictions = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return average_loss, accuracy


def save_checkpoint(path, model, class_names, epoch, val_accuracy, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "config": config,
        },
        path,
    )


def train(config):
    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = create_dataloaders(config)
    print(f"Classes: {class_names}")

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_accuracy = 0.0
    checkpoint_path = config["outputs"]["checkpoint_path"]

    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        train_loss, train_accuracy = run_one_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_accuracy = run_one_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val accuracy:   {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                checkpoint_path,
                model,
                class_names,
                epoch,
                val_accuracy,
                config,
            )
            print(f"Saved best model to {checkpoint_path}")

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a baseline waste classifier.")
    parser.add_argument(
        "--config",
        default="configs/baseline_resnet18.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_config(args.config))
