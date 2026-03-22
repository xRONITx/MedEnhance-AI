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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.enhancer import UNetEnhancer
from utils.metrics import compute_enhancement_metrics
from utils.preprocess import (
    ENHANCER_TRANSFORM,
    degrade_tensor,
    ensure_runtime_directories,
    find_dataset_root,
    list_image_files,
    prepare_xray_image,
)


class EnhancementDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = list(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = prepare_xray_image(self.image_paths[index])
        clean = ENHANCER_TRANSFORM(image)
        degraded = degrade_tensor(clean)
        return degraded, clean


class HybridEnhancementLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, restored, clean):
        return (0.8 * self.l1(restored, clean)) + (0.2 * self.mse(restored, clean))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for degraded, clean in loader:
            degraded = degraded.to(device)
            clean = clean.to(device)
            restored = model(degraded)
            loss = criterion(restored, clean)
            running_loss += loss.item() * degraded.size(0)

            for clean_item, restored_item in zip(clean, restored):
                metrics = compute_enhancement_metrics(
                    clean_item.detach().cpu().permute(1, 2, 0).numpy(),
                    restored_item.detach().cpu().permute(1, 2, 0).numpy(),
                )
                total_psnr += metrics["psnr"]
                total_ssim += metrics["ssim"]

    size = len(loader.dataset)
    return {
        "loss": running_loss / size,
        "psnr": total_psnr / size,
        "ssim": total_ssim / size,
    }


def train(args):
    set_seed(args.seed)
    dataset_root = find_dataset_root(PROJECT_ROOT)
    save_dir = PROJECT_ROOT / "saved_models"
    ensure_runtime_directories([save_dir])
    model_path = save_dir / "enhancer.pth"
    metrics_path = save_dir / "enhancer_metrics.json"

    development_images = list_image_files(dataset_root / "train") + list_image_files(dataset_root / "val")
    train_images, val_images = train_test_split(
        development_images,
        test_size=args.val_fraction,
        random_state=args.seed,
        shuffle=True,
    )

    train_loader = DataLoader(
        EnhancementDataset(train_images),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        EnhancementDataset(val_images),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetEnhancer().to(device)
    criterion = HybridEnhancementLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = float("-inf")
    history = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Enhancer {epoch}/{args.epochs}", leave=False)

        for degraded, clean in progress:
            degraded = degraded.to(device)
            clean = clean.to(device)

            optimizer.zero_grad(set_to_none=True)
            restored = model(degraded)
            loss = criterion(restored, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * degraded.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)
        score = val_metrics["psnr"] + (val_metrics["ssim"] * 100.0)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "score": score,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, indent=2))

        if score > best_score:
            best_score = score
            patience_counter = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "validation_metrics": val_metrics,
                "dataset_root": str(dataset_root),
            }
            torch.save(checkpoint, model_path)
            print(f"Saved improved enhancer checkpoint to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} after {args.patience} epochs without improvement.")
                break

    checkpoint = torch.load(model_path, map_location=device)
    best_model = UNetEnhancer().to(device)
    best_model.load_state_dict(checkpoint["state_dict"])
    final_metrics = evaluate(best_model, val_loader, criterion, device)
    report = {
        "dataset_root": str(dataset_root),
        "model_path": str(model_path),
        "development_split": {
            "train_images": len(train_images),
            "val_images": len(val_images),
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
        "history": history,
        "validation_metrics": final_metrics,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Best validation metrics:")
    print(json.dumps(final_metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the enhancement model with a stronger validation workflow.")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--val-fraction", type=float, default=0.12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
