"""Streamlit app: upload a clothing image and classify it with a trained Fashion-MNIST model.

Run:  streamlit run app.py
"""
import numpy as np
import streamlit as st
from PIL import Image

from src.config import CLASS_NAMES, DEFAULT_MODEL, MODEL_NAMES, MODELS_DIR
from src.preprocess import preprocess

st.set_page_config(page_title="Fashion Classifier", page_icon="👕", layout="centered")

# Custom styling: make dropdown items turn dark with high contrast when selecting/hovered
st.markdown(
    """
    <style>
    div[data-baseweb="popover"] ul[role="listbox"] {
        background-color: #262730 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    div[data-baseweb="popover"] ul[role="listbox"] li[role="option"] {
        background-color: #262730 !important;
        color: #d0d0d0 !important;
    }
    div[data-baseweb="popover"] ul[role="listbox"] li[role="option"] * {
        background-color: transparent !important;
    }
    div[data-baseweb="popover"] ul[role="listbox"] li[role="option"]:hover,
    div[data-baseweb="popover"] ul[role="listbox"] li[role="option"][aria-selected="true"] {
        background-color: #0e1117 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-left: 4px solid #ff4b4b !important;
    }
    div[data-baseweb="popover"] ul[role="listbox"] li[role="option"]:hover *,
    div[data-baseweb="popover"] ul[role="listbox"] li[role="option"][aria-selected="true"] * {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)


def available_models():
    return [k for k in MODEL_NAMES if (MODELS_DIR / f"{k}.keras").exists()]


st.title("👕 Fashion Item Classifier")
st.caption("Upload a picture of a clothing item and the model predicts one of 10 classes.")

keys = available_models()
if not keys:
    st.error("No trained model found. Put `cnn.keras` in the `models/` folder, "
             "or train one with `python -m src.train --model cnn`.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    default_idx = keys.index(DEFAULT_MODEL) if DEFAULT_MODEL in keys else 0
    key = st.selectbox("Model", keys, index=default_idx, format_func=MODEL_NAMES.get)
    auto_invert = st.checkbox("Auto-invert light backgrounds", value=True,
                              help="Fashion-MNIST items are light on a dark background.")
    force_invert = st.checkbox("Always invert", value=False,
                               help="Use if the item is dark on a light background and the result looks wrong.")
    crop = st.checkbox("Crop to the item", value=True)
    st.markdown("---")
    st.markdown("**Classes:** " + ", ".join(CLASS_NAMES))

model = load_model(str(MODELS_DIR / f"{key}.keras"))

file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp", "bmp"])
if file:
    original = Image.open(file)
    processed, batch = preprocess(original, auto_invert, force_invert, crop)

    probs = model.predict(batch, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top = order[0]

    c1, c2 = st.columns(2)
    c1.image(original, caption="Your image", use_container_width=True)
    c2.image(processed.resize((224, 224), Image.NEAREST),
             caption="What the model sees (28×28)", use_container_width=True)

    st.subheader(f"Prediction: {CLASS_NAMES[top]}")
    st.metric("Confidence", f"{probs[top] * 100:.1f}%")
    if probs[top] < 0.6:
        st.info("Low confidence. The item may be outside the 10 classes, "
                "or try the invert/crop options in the sidebar.")

    st.write("**Top 3 predictions**")
    for i in order[:3]:
        st.progress(float(probs[i]), text=f"{CLASS_NAMES[i]}: {probs[i] * 100:.1f}%")

    with st.expander("All class probabilities"):
        st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(10)})
else:
    st.info("👆 Upload an image to get started.")

st.markdown("---")
st.caption("The model knows only 10 clothing classes and was trained on 28×28 images. "
           "It works best on a single item with a plain background.")
