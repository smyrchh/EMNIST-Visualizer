from __future__ import annotations

import base64
import io
import os
from typing import Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms


from training.model_arch import EmnistCNN


def load_model(weights_path: str, device: torch.device) -> EmnistCNN:
    model = EmnistCNN(num_classes=26)
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(
            "Failed to load model weights. The architecture has changed (now Conv2d). "
            "Please retrain with training/train.py to produce a new emnist_model.pth.\n"
            f"Original error: {e}"
        )
    model.to(device)
    model.eval()
    return model


def preprocess_image_from_data_url(data_url: str) -> torch.Tensor:

    
    if "," in data_url:
        _, b64data = data_url.split(",", 1)
    else:
        b64data = data_url

    raw = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(raw)).convert("L")  

   
    img = ImageOps.fit(img, (28, 28), method=Image.Resampling.LANCZOS)

    
    img = ImageOps.invert(img)

    
    img = img.transpose(Image.Transpose.ROTATE_270)
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(img)  
    tensor = tensor.unsqueeze(0)  
    return tensor


def logits_to_prediction(logits: torch.Tensor) -> Tuple[str, float]:
    probs = F.softmax(logits, dim=1)
    conf, idx = torch.max(probs, dim=1)
    idx_int = int(idx.item())
    confidence = float(conf.item())
    letter = chr(ord('A') + idx_int)
    return letter, confidence


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Auto-select the best available device (CUDA -> MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "emnist_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. Train the model to create this file."
        )

    model = load_model(weights_path, device)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(silent=True) or {}
        image_data_url = data.get("image")
        if not image_data_url:
            return jsonify({"error": "Missing 'image' field (data URL)"}), 400

        try:
            tensor = preprocess_image_from_data_url(image_data_url).to(device)
            with torch.no_grad():
                logits = model(tensor)
            letter, confidence = logits_to_prediction(logits)
            return jsonify({
                "prediction": letter,
                "confidence": confidence
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    
    app.run(host="0.0.0.0", port=5001, debug=False)


