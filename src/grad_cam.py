"""
grad_cam.py
-----------
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Highlights which regions of the input image most influenced the model's
prediction, by combining gradients flowing into the last convolutional
layer with that layer's activations.
"""

import io
import base64

import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def overlay_heatmap(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    from PIL import ImageFilter

    size = original_image.size
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(size, resample=Image.BICUBIC)
    cam_img = cam_img.filter(ImageFilter.GaussianBlur(radius=18))
    cam_arr = np.array(cam_img) / 255.0
    cam_arr = np.clip(cam_arr, 0, 1) ** 0.6  # spreads the "hot" region outward

    heatmap = cm.jet(cam_arr)[:, :, :3]
    heatmap = np.uint8(heatmap * 255)
    heatmap_img = Image.fromarray(heatmap).convert("RGB")

    base = original_image.convert("RGB")
    blended = Image.blend(base, heatmap_img, alpha=alpha)
    return blended


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")