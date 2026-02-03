# EMNIST-Visualizer

A clean, minimal **handwritten letter recognition** project built on **PyTorch** and **EMNIST (Letters)** — with training code, a lightweight CNN, and a simple browser-based drawing UI backed by a Flask API.

This repo is intentionally straightforward: no magic, no over-engineering. You can read every file and understand what’s happening.

---

## ✨ What this project does

* Trains a CNN to recognize **handwritten A–Z letters**
* Uses the **EMNIST Letters** dataset (26 classes)
* Exposes a **Flask backend** for inference
* Provides a **canvas-based frontend** to draw letters in your browser
* Saves the **best-performing model** during training

---

## 🧠 Model overview

* Input: `1 × 28 × 28` grayscale image
* Architecture:

  * 3× Conv blocks (32 → 64 → 128 channels)
  * ReLU activations
  * MaxPooling
  * Adaptive average pooling
  * Fully connected classifier (26 classes)
* Loss: Cross‑Entropy
* Optimizer: Adam
* LR Scheduler: StepLR

The architecture is intentionally compact and fast — perfect for real‑time inference.

---

## 📁 Project structure

```
.
├── training/
│   ├── dataset.py        # EMNIST dataloaders
│   ├── model_arch.py     # CNN definition
│   └── train.py          # Training loop
│
├── backend.py             # Flask inference API
├── frontend.py            # Local browser UI (canvas)
├── emnist_model.pth       # Saved model (after training)
└── README.md
```

---

## 🚀 Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Train the model

```bash
cd training
python train.py --epochs 5 --batch_size 64
```

During training:

* Validation accuracy is printed each epoch
* The **best model** is saved as `emnist_model.pth`

---

### 3. Start the backend

From the project root:

```bash
python backend.py
```

The backend:

* Loads `emnist_model.pth`
* Accepts base64 canvas images
* Returns predicted letter + confidence

---

### 4. Launch the frontend

```bash
python frontend.py
```

This opens a local webpage where you can:

* Draw a letter using your mouse / touch
* Send it to the backend
* See the predicted character instantly

---

## 🔄 Inference pipeline 

Canvas → Backend → Model:

1. User draws on a **280×280 canvas**
2. Image is:

   * Converted to grayscale
   * Resized to `28×28`
   * Inverted (white background → black)
   * Rotated & flipped to match EMNIST orientation
3. Normalized and fed to the CNN
4. Softmax → predicted letter + confidence

The preprocessing is **critical** — EMNIST images are not oriented the same way as browser canvas drawings.

---

## 📊 Output format

Backend response:

```json
{
  "letter": "G",
  "confidence": 0.87
}
```
---

Feel free to fork, break, retrain, or extend it.
