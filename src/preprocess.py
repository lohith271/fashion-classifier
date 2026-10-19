"""Turn any user photo into what the model saw in training:
28x28, grayscale, bright item on a dark background, values 0-1.

Only NumPy and Pillow are needed here (no TensorFlow)."""
import numpy as np
from PIL import Image, ImageOps


def preprocess(img: Image.Image, auto_invert: bool = True, force_invert: bool = False, crop: bool = True):
    """Returns (28x28 PIL preview image, model input of shape (1, 28, 28, 1))."""
    img = ImageOps.exif_transpose(img)

    # flatten transparency onto white
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    arr = np.asarray(ImageOps.autocontrast(img.convert("L")), dtype=np.float32)

    # Fashion-MNIST items are bright on dark; photos usually have a bright background.
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]]).mean()
    if force_invert or (auto_invert and border > 127):
        arr = 255.0 - arr

    # crop to the item, then pad to a square with a small margin
    bg_value = float(arr.min())  # background level, taken before cropping
    if crop:
        mask = arr > (arr.min() + 0.25 * (arr.max() - arr.min()))
        if mask.any():
            ys, xs = np.where(mask)
            arr = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = arr.shape
    side = int(max(h, w) * 1.1) + 1
    canvas = np.full((side, side), bg_value, dtype=np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = arr

    small = Image.fromarray(canvas.astype("uint8")).resize((28, 28), Image.LANCZOS)
    x = np.asarray(small, dtype=np.float32) / 255.0
    return small, x[np.newaxis, :, :, np.newaxis]
