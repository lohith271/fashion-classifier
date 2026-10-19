import numpy as np
from PIL import Image, ImageDraw

from src.preprocess import preprocess


def test_output_shape_and_range():
    img = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(img).rectangle([120, 50, 280, 250], fill=(30, 30, 90))
    small, x = preprocess(img)
    assert small.size == (28, 28)
    assert x.shape == (1, 28, 28, 1)
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_light_background_is_inverted():
    img = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(img).rectangle([120, 50, 280, 250], fill=(30, 30, 90))
    _, x = preprocess(img)
    assert x[0, 0, 0, 0] < 0.2      # corner = dark background
    assert x[0, 14, 14, 0] > 0.8    # centre = bright item


def test_dark_background_is_kept():
    img = Image.new("L", (28, 28), 0)
    ImageDraw.Draw(img).rectangle([5, 3, 22, 25], fill=200)
    _, x = preprocess(img)
    assert x[0, 14, 14, 0] > 0.8 and x[0, 0, 0, 0] < 0.2


def test_transparent_png():
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([40, 40, 160, 160], fill=(200, 0, 0, 255))
    _, x = preprocess(img)
    assert x.shape == (1, 28, 28, 1) and np.isfinite(x).all()
