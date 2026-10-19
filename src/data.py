"""Data loading and preprocessing."""
from collections import namedtuple

from sklearn.model_selection import train_test_split
from tensorflow import keras

from .config import SEED, VAL_SIZE

Data = namedtuple("Data", "X_train X_val X_test y_train y_val y_test")


def load_data() -> Data:
    """Load Fashion-MNIST, scale to 0-1, add a channel axis and make a
    stratified 90/10 train/validation split. The test set is left untouched."""
    (X_tr, y_tr), (X_te, y_te) = keras.datasets.fashion_mnist.load_data()

    X_all = X_tr.astype("float32")[..., None] / 255.0
    X_test = X_te.astype("float32")[..., None] / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_tr, test_size=VAL_SIZE, random_state=SEED, stratify=y_tr
    )
    return Data(X_train, X_val, X_test, y_train, y_val, y_te)
