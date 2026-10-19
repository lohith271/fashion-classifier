"""Compare the saved models and analyse the best one.

Run from the project root (after training):
    python -m src.evaluate

The best model is chosen on VALIDATION accuracy. Test accuracy is only reported.

Outputs (results/):
    model_comparison.csv, classification_report.txt, top_confusions.csv
    images/model_comparison.png, confusion_matrix_best.png, misclassified_examples.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow import keras

from .config import CLASS_NAMES, IMAGES_DIR, MODEL_NAMES, MODELS_DIR, RESULTS_DIR
from .data import load_data


def load_saved_models():
    models = {}
    for key in MODEL_NAMES:
        path = MODELS_DIR / f"{key}.keras"
        if path.exists():
            models[key] = keras.models.load_model(path, compile=False)
    return models


def compare(models, data):
    rows = []
    for key, model in models.items():
        model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        _, val_acc = model.evaluate(data.X_val, data.y_val, verbose=0)
        test_loss, test_acc = model.evaluate(data.X_test, data.y_test, verbose=0)

        meta_file = RESULTS_DIR / f"metrics_{key}.json"
        train_time = json.loads(meta_file.read_text()).get("train_time_s") if meta_file.exists() else None
        rows.append({
            "Key": key, "Model": MODEL_NAMES[key],
            "Val Accuracy (%)": round(val_acc * 100, 2),
            "Test Accuracy (%)": round(test_acc * 100, 2),
            "Test Loss": round(test_loss, 4),
            "Parameters": model.count_params(),
            "Train Time (s)": train_time,
        })
    return pd.DataFrame(rows).sort_values("Val Accuracy (%)", ascending=False).reset_index(drop=True)


def plot_comparison(df, path):
    plot_df = df.sort_values("Test Accuracy (%)")
    plt.figure(figsize=(9, 4))
    bars = plt.barh(plot_df["Model"], plot_df["Test Accuracy (%)"], color="steelblue")
    plt.bar_label(bars, fmt="%.2f", padding=3)
    plt.xlim(plot_df["Test Accuracy (%)"].min() - 5, 100)
    plt.xlabel("Test accuracy (%)")
    plt.title("Fashion-MNIST: model comparison")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def analyse_best(model, name, data):
    y_pred = np.argmax(model.predict(data.X_test, verbose=0), axis=1)
    y_true = data.y_test

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    (RESULTS_DIR / "classification_report.txt").write_text(f"Best model: {name}\n\n{report}")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion matrix: {name}")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "confusion_matrix_best.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(CLASS_NAMES[i], CLASS_NAMES[j], int(off[i, j]))
             for i in range(10) for j in range(10) if off[i, j] > 0]
    top = pd.DataFrame(pairs, columns=["True class", "Predicted as", "Count"]).sort_values("Count", ascending=False).head(8)
    top.to_csv(RESULTS_DIR / "top_confusions.csv", index=False)
    print(top.to_string(index=False))

    wrong = np.where(y_pred != y_true)[0]
    print(f"\nMisclassified: {len(wrong)} of {len(y_true)}")
    plt.figure(figsize=(12, 9))
    for i, idx in enumerate(wrong[:12]):
        plt.subplot(3, 4, i + 1)
        plt.imshow(data.X_test[idx].squeeze(), cmap="gray")
        plt.title(f"True: {CLASS_NAMES[y_true[idx]]}\nPred: {CLASS_NAMES[y_pred[idx]]}", fontsize=9)
        plt.axis("off")
    plt.suptitle("Examples of misclassified images")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "misclassified_examples.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    models = load_saved_models()
    if not models:
        raise SystemExit("No models found in models/. Train first: python -m src.train --model cnn")

    data = load_data()
    df = compare(models, data)
    best_key = df.loc[0, "Key"]
    print(f"Best model (by validation accuracy): {MODEL_NAMES[best_key]}\n")
    print(df.drop(columns="Key").to_string(index=False), "\n")

    df.drop(columns="Key").to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    plot_comparison(df, IMAGES_DIR / "model_comparison.png")
    analyse_best(models[best_key], MODEL_NAMES[best_key], data)
    print("\nSaved results to results/")


if __name__ == "__main__":
    main()
