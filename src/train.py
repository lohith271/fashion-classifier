"""Train ANN / CNN / ResNet50 on Fashion-MNIST.

Run from the project root:
    python -m src.train --model cnn
    python -m src.train --model ann --tune-dropout
    python -m src.train --model all

Outputs:
    models/<key>.keras                 trained model (ResNet50 only with --save-resnet)
    results/metrics_<key>.json         accuracy, parameters, training time
    results/images/<key>_history.png   training curves
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from .config import IMAGES_DIR, MODEL_NAMES, MODELS_DIR, RESULTS_DIR, SEED
from .data import load_data
from .models import build_ann, build_cnn, build_resnet50, unfreeze_top

# One shared recipe: Adam, early stopping on val loss (restore best weights),
# learning rate halved when val loss stalls.
RECIPES = {
    "ann": dict(epochs=40, batch_size=128, patience=6, lr_patience=2),
    "cnn": dict(epochs=40, batch_size=128, patience=8, lr_patience=3),
    "resnet50_stage1": dict(epochs=15, batch_size=64, patience=3, lr_patience=2),
    "resnet50_finetune": dict(epochs=10, batch_size=64, patience=3, lr_patience=2),
}


def fit(model, data, epochs, batch_size, patience, lr_patience, verbose=2):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=lr_patience, min_lr=1e-6),
    ]
    start = time.time()
    history = model.fit(
        data.X_train, data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose,
    )
    return history.history, time.time() - start


def score(model, data):
    _, val_acc = model.evaluate(data.X_val, data.y_val, verbose=0)
    test_loss, test_acc = model.evaluate(data.X_test, data.y_test, verbose=0)
    return val_acc, test_acc, test_loss


def plot_history(history, name, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history[metric], label="Train")
        ax.plot(history["val_" + metric], label="Validation")
        ax.set(title=f"{name}: {metric}", xlabel="Epoch", ylabel=metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- trainers
# Each returns (model, {history_name: history}, train_time_seconds, extra_info)
def train_ann(data, args):
    rate = 0.3
    if args.tune_dropout:
        scores = {}
        for r in (0.0, 0.2, 0.3):
            keras.utils.set_random_seed(SEED)
            m = build_ann(r)
            fit(m, data, verbose=0, **RECIPES["ann"])
            scores[r] = score(m, data)[0]
            print(f"  dropout {r}: val acc {scores[r] * 100:.2f}%")
        rate = max(scores, key=scores.get)
        print(f"Best dropout (validation): {rate}")

    keras.utils.set_random_seed(SEED)
    model = build_ann(rate)
    hist, t = fit(model, data, **RECIPES["ann"])
    return model, {"ann": hist}, t, {"dropout": rate}


def train_cnn(data, args):
    keras.utils.set_random_seed(SEED)
    model = build_cnn()
    hist, t = fit(model, data, **RECIPES["cnn"])
    return model, {"cnn": hist}, t, {}


def train_resnet50(data, args):
    keras.utils.set_random_seed(SEED)
    model, base = build_resnet50()
    hist1, t1 = fit(model, data, **RECIPES["resnet50_stage1"])
    histories, total = {"resnet50_stage1": hist1}, t1
    if not args.no_finetune:
        unfreeze_top(model, base)
        hist2, t2 = fit(model, data, **RECIPES["resnet50_finetune"])
        histories["resnet50"] = hist2
        total += t2
    return model, histories, total, {"fine_tuned": not args.no_finetune}


TRAINERS = {"ann": train_ann, "cnn": train_cnn, "resnet50": train_resnet50}


def main():
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST models")
    parser.add_argument("--model", choices=[*TRAINERS, "all"], default="cnn")
    parser.add_argument("--tune-dropout", action="store_true", help="ANN: pick dropout on validation data")
    parser.add_argument("--no-finetune", action="store_true", help="ResNet50: stop after the frozen stage")
    parser.add_argument("--save-resnet", action="store_true", help="also save the (large) ResNet50 model")
    args = parser.parse_args()

    for d in (MODELS_DIR, RESULTS_DIR, IMAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    data = load_data()
    print(f"Train {data.X_train.shape} | Val {data.X_val.shape} | Test {data.X_test.shape}")

    keys = list(TRAINERS) if args.model == "all" else [args.model]
    for key in keys:
        print(f"\n=== Training {MODEL_NAMES[key]} ===")
        model, histories, train_time, extra = TRAINERS[key](data, args)

        val_acc, test_acc, test_loss = score(model, data)
        metrics = {
            "key": key, "model": MODEL_NAMES[key],
            "val_accuracy": round(val_acc * 100, 2),
            "test_accuracy": round(test_acc * 100, 2),
            "test_loss": round(test_loss, 4),
            "parameters": int(model.count_params()),
            "train_time_s": round(train_time),
            **extra,
        }
        print(f"{MODEL_NAMES[key]}: val {metrics['val_accuracy']}% | test {metrics['test_accuracy']}%")

        (RESULTS_DIR / f"metrics_{key}.json").write_text(json.dumps(metrics, indent=2))
        for name, hist in histories.items():
            plot_history(hist, name, IMAGES_DIR / f"{name}_history.png")

        if key != "resnet50" or args.save_resnet:
            model.save(MODELS_DIR / f"{key}.keras")
            print(f"Saved models/{key}.keras")

    print("\nDone. Next: python -m src.evaluate")


if __name__ == "__main__":
    main()
