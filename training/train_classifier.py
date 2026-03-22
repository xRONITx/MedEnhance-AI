import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.classifier import get_classifier
from utils.metrics import compute_metrics_from_probabilities, find_best_classification_threshold
from utils.preprocess import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, find_dataset_root, prepare_xray_image

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


class ChestXrayDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = prepare_xray_image(image_path)
        return self.transform(image), label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_split_samples(dataset_root, splits):
    samples = []
    for split in splits:
        for label_index, class_name in enumerate(CLASS_NAMES):
            folder = dataset_root / split / class_name
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    samples.append((path, label_index))
    return samples


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.08, contrast=0.12),
            transforms.RandomAdjustSharpness(sharpness_factor=1.15, p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def create_balanced_sampler(samples):
    labels = [label for _, label in samples]
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    sample_weights = [float(class_weights[label]) for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler, class_counts


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_true = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            running_loss += loss.item() * images.size(0)
            all_true.extend(labels.cpu().tolist())
            all_probs.extend(probabilities.cpu().tolist())

    metrics = compute_metrics_from_probabilities(all_true, all_probs, threshold=threshold)
    metrics["loss"] = running_loss / len(loader.dataset)
    metrics["positive_probabilities"] = all_probs
    metrics["targets"] = all_true
    return metrics


def train(args):
    set_seed(args.seed)
    dataset_root = find_dataset_root(PROJECT_ROOT)
    save_dir = PROJECT_ROOT / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "classifier.pth"
    metrics_path = save_dir / "classifier_metrics.json"

    development_samples = collect_split_samples(dataset_root, splits=("train", "val"))
    test_samples = collect_split_samples(dataset_root, splits=("test",))
    labels = [label for _, label in development_samples]

    train_samples, val_samples = train_test_split(
        development_samples,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=labels,
    )

    train_transform, eval_transform = build_transforms()
    train_dataset = ChestXrayDataset(train_samples, transform=train_transform)
    val_dataset = ChestXrayDataset(val_samples, transform=eval_transform)
    test_dataset = ChestXrayDataset(test_samples, transform=eval_transform)

    train_sampler, class_counts = create_balanced_sampler(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_classifier(num_classes=2, use_pretrained=not args.no_pretrained).to(device)

    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32, device=device)
    try:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("fc"):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.learning_rate * 0.35},
            {"params": head_params, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.learning_rate * 0.35, args.learning_rate],
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        div_factor=25.0,
        final_div_factor=100.0,
    )

    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_score = float("-inf")
    best_threshold = 0.5
    best_epoch = 0
    history = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)
        val_base = evaluate(model, val_loader, criterion, device, threshold=0.5)
        tuned_threshold, val_metrics = find_best_classification_threshold(
            val_base["targets"],
            val_base["positive_probabilities"],
            step=0.01,
        )
        val_metrics["loss"] = val_base["loss"]

        score = (val_metrics["f1"] * 0.6) + (val_metrics["recall"] * 0.3) + (val_metrics["accuracy"] * 0.1)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "threshold": tuned_threshold,
            "score": score,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, indent=2))

        if score > best_score:
            best_score = score
            best_threshold = tuned_threshold
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "threshold": best_threshold,
                "classes": CLASS_NAMES,
                "dropout_rate": 0.35,
                "best_epoch": best_epoch,
                "val_metrics": val_metrics,
                "dataset_root": str(dataset_root),
            }
            torch.save(checkpoint, model_path)
            print(f"Saved improved classifier checkpoint to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} after {args.patience} epochs without improvement.")
                break

    checkpoint = torch.load(model_path, map_location=device)
    best_model = get_classifier(num_classes=2, use_pretrained=False).to(device)
    best_model.load_state_dict(checkpoint["state_dict"])
    best_threshold = float(checkpoint.get("threshold", 0.5))

    test_metrics = evaluate(best_model, test_loader, criterion, device, threshold=best_threshold)
    report = {
        "dataset_root": str(dataset_root),
        "model_path": str(model_path),
        "development_split": {
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "history": history,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "loss": test_metrics["loss"],
        },
        "classes": CLASS_NAMES,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Final test metrics:")
    print(json.dumps(report["test_metrics"], indent=2))
    print(f"Best decision threshold: {best_threshold:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pneumonia classifier with a more robust pipeline.")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
