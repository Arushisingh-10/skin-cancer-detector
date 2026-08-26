"""
train.py
--------
Model training script. This script:
  1. Loads the dataset and splits it into training/validation sets
  2. Applies data augmentation for the training set
  3. Trains the model and saves the best checkpoint
  4. Handles class imbalance (the 'nv' class heavily dominates HAM10000,
     so a weighted loss function is used)

Usage:
    python src/train.py --csv data/HAM10000_metadata.csv \
                         --img_dirs data/HAM10000_images_part_1 data/HAM10000_images_part_2 \
                         --epochs 15 --batch_size 32
"""

import argparse
import os
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

from dataset import HAM10000Dataset, CLASS_NAMES
from model import build_model


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def compute_class_weights(dataset):
    """Compute per-class weights to counteract class imbalance (the 'nv'
    class dominates the dataset)."""
    labels = dataset.df["dx"].map({name: i for i, name in enumerate(CLASS_NAMES)}).values
    weights = compute_class_weight(class_weight="balanced",
                                    classes=np.arange(len(CLASS_NAMES)),
                                    y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, val_tf = get_transforms()

    full_dataset = HAM10000Dataset(args.csv, args.img_dirs, transform=None)
    class_weights = compute_class_weights(full_dataset).to(device)

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    # Assign separate transforms after the split
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=args.freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # ---- Training phase ----
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # ---- Validation phase ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  -> New best model found (Val Acc: {val_acc:.4f})")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "skin_cancer_model.pth")
    torch.save({
        "model_state_dict": best_model_wts,
        "class_names": CLASS_NAMES,
    }, save_path)
    print(f"\nBest model saved to: {save_path} (Val Acc: {best_acc:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skin Cancer Detector - Training")
    parser.add_argument("--csv", type=str, required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--img_dirs", type=str, nargs="+", required=True, help="Image folder paths")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--save_dir", type=str, default="../saved_models")
    args = parser.parse_args()

    train_model(args)