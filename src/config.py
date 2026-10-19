"""Shared constants and paths (no TensorFlow import, so the app can load it fast)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = RESULTS_DIR / "images"

SEED = 42
VAL_SIZE = 0.1
IMG_SHAPE = (28, 28, 1)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# key -> display name
MODEL_NAMES = {"ann": "ANN", "cnn": "CNN", "resnet50": "ResNet50"}

# model used by the Streamlit app by default (best on validation accuracy in the notebook)
DEFAULT_MODEL = "cnn"
