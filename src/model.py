from torch import nn
from torchvision import models


def build_resnet18(num_classes, pretrained=True):
    """Build a ResNet18 classifier for the requested number of classes."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(config):
    """Build the model selected in the config file."""
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"].get("pretrained", True)

    if model_name != "resnet18":
        raise ValueError(f"Unsupported model: {model_name}")

    return build_resnet18(num_classes=num_classes, pretrained=pretrained)
