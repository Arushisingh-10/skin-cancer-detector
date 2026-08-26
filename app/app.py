"""
app.py
------
Flask backend that:
  1. Loads the trained model at startup
  2. Serves the upload page on the '/' route
  3. Accepts an image on '/predict' and returns the prediction as JSON

Usage (run from inside the skin_cancer_detector/app folder):
    python app.py

Then open: http://127.0.0.1:5000
"""

import os
import sys

from flask import Flask, request, jsonify, render_template
from PIL import Image

# Add the src/ folder to the path so predict.py can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import load_model, predict_image, is_likely_skin_image  # noqa: E402

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models", "skin_cancer_model.pth")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file found. Send it under the 'image' field."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PNG/JPG/JPEG images are allowed."}), 400

    try:
        image = Image.open(file.stream)

        if not is_likely_skin_image(image):
            return jsonify({
                "error": "This doesn't look like a valid skin lesion image. Please upload a close-up photo of a skin lesion or mole."
            }), 400

        result = predict_image(image)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at: {MODEL_PATH}")
        print("Run 'src/train.py' first to train a model.")
    else:
        load_model(MODEL_PATH)
        print("Model loaded successfully!")

    app.run(debug=True, host="0.0.0.0", port=5000)