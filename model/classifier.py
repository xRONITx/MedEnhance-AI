from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def get_resnet18_backbone(use_pretrained=True):
    try:
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        return models.resnet18(weights=weights)
    except AttributeError:
        return models.resnet18(pretrained=use_pretrained)


def _build_classifier_head(in_features, num_classes=2, dropout_rate=0.35, legacy_head=False):
    if legacy_head:
        return nn.Linear(in_features, num_classes)
    return nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )


def get_classifier(num_classes=2, use_pretrained=True, dropout_rate=0.35, legacy_head=False):
    model = get_resnet18_backbone(use_pretrained=use_pretrained)
    model.fc = _build_classifier_head(
        model.fc.in_features,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        legacy_head=legacy_head,
    )
    return model


def _unpack_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    return checkpoint, {}


def _uses_legacy_head(state_dict):
    return "fc.weight" in state_dict or "fc.bias" in state_dict


def load_classifier(model_path, device):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier weights not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict, metadata = _unpack_checkpoint(checkpoint)
    legacy_head = _uses_legacy_head(state_dict)

    model = get_classifier(
        num_classes=2,
        use_pretrained=False,
        dropout_rate=float(metadata.get("dropout_rate", 0.35)),
        legacy_head=legacy_head,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    threshold = float(metadata.get("threshold", 0.5))
    classes = metadata.get("classes", ["NORMAL", "PNEUMONIA"])
    target_layer = model.layer4[-1].conv2
    info = {
        "threshold": threshold,
        "classes": classes,
        "legacy_head": legacy_head,
    }
    return model, target_layer, info
