# Skin Cancer Detector (Deep Learning)

A skin lesion classifier built with transfer learning (EfficientNet-B0) on
the HAM10000 dataset, paired with a Flask web application for image upload
and real-time prediction.

Classifies images into 7 lesion categories:

| Code  | Full Name                         | Type       |
|-------|-----------------------------------|------------|
| akiec | Actinic Keratoses                 | Malignant* |
| bcc   | Basal Cell Carcinoma              | Malignant  |
| bkl   | Benign Keratosis-like Lesion      | Benign     |
| df    | Dermatofibroma                    | Benign     |
| mel   | **Melanoma**                      | Malignant  |
| nv    | Melanocytic Nevi (mole)           | Benign     |
| vasc  | Vascular Lesion                   | Benign     |

*akiec is pre-cancerous but is grouped with the malignant-risk classes here.

---

## ⚠️ Disclaimer

This project is intended for **educational and portfolio purposes only**.
It is not a certified medical diagnostic tool and must not be used for
real patient diagnosis. Consult a qualified dermatologist for any skin
health concerns.

---

## Project Structure

```
skin_cancer_detector/
├── requirements.txt
├── data/                          # dataset goes here (see instructions below)
├── saved_models/                  # trained model checkpoint is saved here
├── src/
│   ├── dataset.py                 # PyTorch Dataset class
│   ├── model.py                   # EfficientNet-B0 based model
│   ├── train.py                   # training script
│   ├── evaluate.py                # evaluation + confusion matrix
│   └── predict.py                 # single-image inference helper
└── app/
    ├── app.py                     # Flask backend
    ├── templates/index.html       # upload UI
    └── static/style.css
```

---

## Architecture Overview

- **Model:** EfficientNet-B0 (Convolutional Neural Network), pretrained on
  ImageNet, fine-tuned via transfer learning
- **Framework:** PyTorch
- **Input:** 224×224 RGB dermatoscopic images
- **Output:** Softmax probability distribution over 7 lesion classes
- **Class imbalance handling:** Weighted cross-entropy loss
  (the `nv` class heavily dominates the raw dataset)

---

## Step 1: Environment Setup

```bash
cd skin_cancer_detector
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 2: Dataset Download (HAM10000)

The dataset is freely available from two sources:

**Option A — Kaggle (recommended)**
1. Visit https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Click "Download" (requires a free Kaggle account)
3. Extract the ZIP into the `data/` folder

**Option B — Official ISIC / Harvard Dataverse source**
1. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
2. Download the original HAM10000 files from here

After extraction, `data/` should look like this:

```
data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
│   ├── ISIC_0024306.jpg
│   └── ...
└── HAM10000_images_part_2/
    ├── ISIC_0029306.jpg
    └── ...
```

(Some Kaggle versions place all images in a single folder — in that case,
pass just that one path to `--img_dirs`.)

---

## Step 3: Train the Model

```bash
cd src
python train.py \
    --csv ../data/HAM10000_metadata.csv \
    --img_dirs ../data/HAM10000_images_part_1 ../data/HAM10000_images_part_2 \
    --epochs 15 \
    --batch_size 32
```

Notes:
- Uses CUDA automatically if a GPU is available
- Also runs on CPU, though considerably slower
- The best checkpoint is saved to `saved_models/skin_cancer_model.pth`
- Class imbalance is handled automatically via weighted loss

**Tips if training is slow:**
- Reduce `--batch_size` if you hit an out-of-memory error
- Try `--epochs 5` for a quick smoke test before running the full job
- Google Colab (free GPU) works well with this code — upload the project
  and run the same command there

---

## Step 4: Evaluate the Model

```bash
python evaluate.py \
    --csv ../data/HAM10000_metadata.csv \
    --img_dirs ../data/HAM10000_images_part_1 ../data/HAM10000_images_part_2 \
    --model_path ../saved_models/skin_cancer_model.pth
```

This prints a classification report (precision/recall/F1) and saves
`confusion_matrix.png`.

---

## Step 5: Run the Web App

```bash
cd ../app
python app.py
```

Open your browser at **http://127.0.0.1:5000**.

Upload an image (drag-and-drop or click) → click "Analyze Lesion" → view
the prediction with per-class confidence scores.

---

## Deployment (Production-Ready Steps)

1. **Serve with Gunicorn** (instead of the Flask dev server):
   ```bash
   pip install gunicorn
   gunicorn --bind 0.0.0.0:8000 app:app
   ```
2. **Docker**, if preferred:
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt gunicorn
   COPY . .
   CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app.app:app"]
   ```
3. Cloud deployment: Render, Railway, Hugging Face Spaces, or AWS
   EC2/Elastic Beanstalk are all directly compatible with this Flask
   structure.
4. **The model file is fairly large** — use Git LFS or cloud storage (S3)
   rather than committing it directly to a GitHub repository.

---

## Possible Extensions

- Set `freeze_backbone=False` to try full fine-tuning (higher accuracy,
  longer training time)
- Compare EfficientNet-B0 against ResNet50 / EfficientNet-B4
- Add Grad-CAM visualizations to show which regions of the image drove
  the prediction
- Add test-time augmentation to improve confidence estimates
- Enable binary mode (`binary=True` in `dataset.py`) for a simpler,
  more robust malignant-vs-benign classifier
