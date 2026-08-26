"""
predict.py
----------
Helper module for running inference on a single image.
This module is imported by the Flask app to serve predictions.
"""
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import build_model
from grad_cam import GradCAM, overlay_heatmap, image_to_base64

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratoses (pre-cancerous)",
    "bcc": "Basal Cell Carcinoma (malignant)",
    "bkl": "Benign Keratosis-like Lesion (benign)",
    "df": "Dermatofibroma (benign)",
    "mel": "Melanoma (malignant - most dangerous)",
    "nv": "Melanocytic Nevi / Mole (benign)",
    "vasc": "Vascular Lesion (benign)",
}

MALIGNANT_CLASSES = {"akiec", "bcc", "mel"}

def is_likely_skin_image(image: Image.Image, min_skin_ratio: float = 0.35) -> bool:
    """Lightweight sanity check: returns True if enough of the image's
    pixels fall within a typical skin-tone RGB range. This rejects
    obviously unrelated photos (screenshots, landscapes, objects, text)
    but is not a full validity classifier."""
    img = image.convert("RGB").resize((100, 100))
    arr = np.array(img).astype(np.int32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)

    skin_mask = (
        (r > 95) & (g > 40) & (b > 20) &
        ((max_c - min_c) > 15) &
        (np.abs(r - g) > 15) &
        (r > g) & (r > b)
    )

    skin_ratio = skin_mask.sum() / skin_mask.size
    return skin_ratio >= min_skin_ratio

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

_model = None
_gradcam = None

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    """Load the trained model checkpoint into memory."""
    
    global _model, _gradcam
    checkpoint = torch.load(model_path, map_location=_device)
    _model = build_model(num_classes=len(CLASS_NAMES))
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(_device)
    _model.eval()

    target_layer = _model.features[-1]
    _gradcam = GradCAM(_model, target_layer)
    return _model


def predict_image(image: Image.Image):
    """
    Run inference on a single PIL image.

    Args:
        image (PIL.Image): Input skin lesion image
    Returns:
        dict containing the predicted class, its full name, a malignancy
        flag, and confidence scores for all 7 classes
    """
    if _model is None:
        raise RuntimeError("Model has not been loaded. Call load_model() first.")

    image = image.convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(_device)
    tensor.requires_grad_(True)

    with torch.no_grad():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = probs.argmax()
    pred_class = CLASS_NAMES[pred_idx]

    
    # Generate Grad-CAM heatmap showing which regions influenced the prediction
    resized_image = image.resize((224, 224))
    cam = _gradcam.generate(tensor, pred_idx)
    overlay = overlay_heatmap(resized_image, cam)
    heatmap_b64 = image_to_base64(overlay)

    return {
        "predicted_class": pred_class,
        "predicted_full_name": CLASS_FULL_NAMES[pred_class],
        "is_malignant_risk": pred_class in MALIGNANT_CLASSES,
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
        "heatmap": heatmap_b64,
    }