"""
evaluate.py
-----------
Evaluates a trained model on the validation split.
Prints a classification report (precision/recall/F1) and saves a
confusion matrix plot.

Usage:
    python src/evaluate.py --csv data/HAM10000_metadata.csv \
                            --img_dirs data/HAM10000_images_part_1 data/HAM10000_images_part_2 \
                            --model_path ../saved_models/skin_cancer_model.pth
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import HAM10000Dataset, CLASS_NAMES
from model import build_model


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = HAM10000Dataset(args.csv, args.img_dirs, transform=val_tf)
    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size
    _, val_ds = random_split(full_dataset, [train_size, val_size],
                              generator=torch.Generator().manual_seed(42))

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    checkpoint = torch.load(args.model_path, map_location=device)
    model = build_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Skin Cancer Classifier")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\nConfusion matrix saved to: confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--img_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    evaluate(args)