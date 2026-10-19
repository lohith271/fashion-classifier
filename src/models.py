"""Model definitions: ANN, CNN and ResNet50 (transfer learning).

Every model takes (28, 28, 1) images scaled to 0-1 and outputs 10 softmax probabilities.
"""
from tensorflow import keras
from tensorflow.keras import layers

from .config import IMG_SHAPE


def compile_model(model, lr: float = 1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------------------------------------------------ ANN
def _dense_block(units, dropout):
    return [
        layers.Dense(units, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(dropout),
    ]


def build_ann(dropout: float = 0.3):
    """Fully connected baseline: 512-256-128 units, batch norm and dropout."""
    model = keras.Sequential([
        keras.Input(shape=IMG_SHAPE),
        layers.Flatten(),
        *_dense_block(512, dropout),
        *_dense_block(256, dropout),
        *_dense_block(128, dropout / 2),
        layers.Dense(10, activation="softmax"),
    ], name="ann")
    return compile_model(model)


# ------------------------------------------------------------------ CNN
def _conv_block(x, filters, dropout):
    for _ in range(2):
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return layers.Dropout(dropout)(x)


def build_cnn():
    """3 conv blocks (32/64/128), augmentation (training only), global average pooling."""
    inputs = keras.Input(shape=IMG_SHAPE)

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomRotation(0.05)(x)

    x = _conv_block(x, 32, 0.25)
    x = _conv_block(x, 64, 0.25)
    x = _conv_block(x, 128, 0.30)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    return compile_model(keras.Model(inputs, outputs, name="cnn"))


# ------------------------------------------------------------- ResNet50
def build_resnet50(pretrained: bool = True):
    """ImageNet ResNet50 on upscaled images. Returns (model, backbone).

    Inside the model: resize 28->96, repeat grayscale to 3 channels,
    rescale to 0-255 and subtract the ImageNet channel means.
    The backbone starts frozen; call `unfreeze_top` for fine-tuning."""
    base = keras.applications.ResNet50(
        weights="imagenet" if pretrained else None,
        include_top=False,
        input_shape=(96, 96, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=IMG_SHAPE)
    x = layers.Resizing(96, 96)(inputs)
    x = layers.Concatenate()([x, x, x])
    x = layers.Rescaling(255.0)(x)
    x = layers.Normalization(mean=[103.939, 116.779, 123.68], variance=[1.0, 1.0, 1.0])(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="resnet50_transfer")
    return compile_model(model), base


def unfreeze_top(model, base, n_layers: int = 25, lr: float = 1e-5):
    """Fine-tune: unfreeze the top `n_layers` of the backbone, use a tiny learning rate."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    return compile_model(model, lr=lr)
