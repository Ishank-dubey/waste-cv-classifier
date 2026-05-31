from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


EXPECTED_CLASSES = [
    "organic",
    "plastic",
    "paper_cardboard",
    "metal",
    "glass",
    "other",
]


def build_transforms(image_size):
    """Create training and evaluation image transforms."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, eval_transform


def load_image_folder(data_dir, transform):
    """Load one ImageFolder split and give a helpful error if it is missing."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}. "
            "Create the ImageFolder structure described in README.md."
        )

    return datasets.ImageFolder(root=data_dir, transform=transform)


def create_dataloaders(config):
    """Create train and validation dataloaders."""
    data_config = config["data"]
    train_transform, eval_transform = build_transforms(data_config["image_size"])

    train_dataset = load_image_folder(data_config["train_dir"], train_transform)
    val_dataset = load_image_folder(data_config["val_dir"], eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    return train_loader, val_loader, train_dataset.classes


def create_test_loader(config):
    """Create the test dataloader."""
    data_config = config["data"]
    _, eval_transform = build_transforms(data_config["image_size"])
    test_dataset = load_image_folder(data_config["test_dir"], eval_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    return test_loader, test_dataset.classes
