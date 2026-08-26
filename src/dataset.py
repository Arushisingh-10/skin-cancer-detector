"""
dataset.py
----------
PyTorch Dataset class for the HAM10000 skin lesion dataset.

HAM10000 metadata CSV columns (Kaggle version):
    lesion_id, image_id, dx, dx_type, age, sex, localization

The 'dx' column contains 7 diagnostic categories:
    akiec - Actinic keratoses / intraepithelial carcinoma
    bcc   - Basal cell carcinoma
    bkl   - Benign keratosis-like lesions
    df    - Dermatofibroma
    mel   - Melanoma (malignant)
    nv    - Melanocytic nevi (common mole, benign)
    vasc  - Vascular lesions

This module:
  1. Reads the metadata CSV
  2. Resolves image file paths (images may be split across two folders,
     e.g. HAM10000_images_part_1 and part_2, so both are checked)
  3. Encodes labels as integers for training
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# 7-class label mapping (kept in alphabetical order for consistency)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Malignant vs. benign grouping (used when binary classification is enabled)
MALIGNANT_CLASSES = {"akiec", "bcc", "mel"}


class HAM10000Dataset(Dataset):
    def __init__(self, csv_path, img_dirs, transform=None, binary=False):
        """
        Args:
            csv_path (str): Path to the metadata CSV (e.g. HAM10000_metadata.csv)
            img_dirs (list[str]): List of image directories to search
            transform: torchvision transforms to apply to each image
            binary (bool): If True, labels are 0 (benign) / 1 (malignant)
                            instead of the full 7-class label
        """
        self.df = pd.read_csv(csv_path)
        self.img_dirs = img_dirs
        self.transform = transform
        self.binary = binary

        # Build a lookup of image_id -> file path across all provided directories
        self.image_paths = {}
        for d in self.img_dirs:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_id = os.path.splitext(fname)[0]
                    self.image_paths[image_id] = os.path.join(d, fname)

        # Keep only rows for which the corresponding image file was found
        self.df = self.df[self.df["image_id"].isin(self.image_paths.keys())].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                "No images were found. Please check img_dirs — "
                "has the dataset been extracted to the expected location?"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        dx = row["dx"]

        img_path = self.image_paths[image_id]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.binary:
            label = 1 if dx in MALIGNANT_CLASSES else 0
        else:
            label = CLASS_TO_IDX[dx]

        return image, label