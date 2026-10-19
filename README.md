# Fashion-MNIST Image Classification: ANN vs CNN vs ResNet50

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fashion-classifier-5esprsdq2sikwhbijfuzsf.streamlit.app/)
**Live Demo:** [fashion-classifier-5esprsdq2sikwhbijfuzsf.streamlit.app](https://fashion-classifier-5esprsdq2sikwhbijfuzsf.streamlit.app/)

Classifies 28×28 grayscale clothing images into 10 classes and compares three model families,
each tuned with techniques that suit it. The best model (a CNN) is served in a **Streamlit app**
where anyone can upload an image and get a prediction.

## Results

All models use the same data, seed and training recipe. A stratified 90/10 split of the 60,000
training images gives train/validation sets; the best model is chosen on **validation** accuracy and
the 10,000 test images are only used for reporting.

| Model | Test accuracy | Parameters | Techniques |
|---|---|---|---|
| ANN | 90.33% | ~570k | Batch norm, dropout (rate tuned on validation data), early stopping, LR scheduling |
| **CNN (best)** | **92.25%** (92.48 ± 0.15% over 3 seeds) | ~305k | 3 conv blocks, batch norm, data augmentation, global average pooling, dropout |
| ResNet50 | 90.25% | ~23.8M | ImageNet preprocessing, frozen backbone then fine-tuning the top 25 layers |

Key findings
- The CNN beats the ANN by about 2 points with fewer parameters, because convolutions use the 2D structure of the image.
- Fine-tuned ResNet50 did not beat the CNN: ImageNet features suit large colour photos, not upscaled 28×28 grayscale images. Fine-tuning added only ~0.2 points and overfit.
- Hardest classes: Shirt (F1 0.77), then T-shirt/top and Coat. Most common error: T-shirt/top predicted as Shirt.

Plots and tables are in [`results/`](results/) after you run `python -m src.evaluate`.

## Project structure

```
fashion-mnist-classifier/
├── app.py                  # Streamlit web app
├── src/
│   ├── config.py           # paths, constants, class names
│   ├── data.py             # load + split Fashion-MNIST
│   ├── models.py           # ANN, CNN, ResNet50 definitions
│   ├── train.py            # training script (CLI)
│   ├── evaluate.py         # comparison, confusion matrix, error analysis
│   └── preprocess.py       # turns any uploaded photo into a model input
├── models/                 # saved models (cnn.keras, ann.keras)
├── results/                # metrics, reports and plots
│   └── images/
├── notebooks/
│   └── Fashion_MNIST.ipynb # interactive exploration, training & export
├── tests/
│   └── test_preprocess.py
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
```

### 1. Training the Models (Choose CLI or Notebook)

* **Option A: Via Jupyter Notebook**
  Open [`notebooks/Fashion_MNIST.ipynb`](notebooks/Fashion_MNIST.ipynb). Running the cells sequentially will perform EDA, data preprocessing, train the ANN, CNN, and ResNet50 models, and automatically save the trained `.keras` model weights into the `models/` directory.

* **Option B: Via Command Line**
  ```bash
  python -m src.train --model cnn                 # or: ann, resnet50, all
  python -m src.train --model ann --tune-dropout  # pick ANN dropout on validation data
  ```

### 2. Compare models and analyse the best one
```bash
python -m src.evaluate
```

### 3. Launch the Streamlit app locally
```bash
streamlit run app.py
```

### 4. Run tests
```bash
pytest
```

`resnet50` is not saved by default because the file is large; add `--save-resnet` to keep it.

## How the app handles real photos

The model was trained on 28×28 grayscale images with a bright item on a dark background.
`src/preprocess.py` therefore converts uploads to grayscale, inverts light backgrounds, crops to the
item, pads to a square and resizes to 28×28 before predicting. The app shows the 28×28 image the
model sees, the top 3 predictions and a warning when confidence is low.

Limitations: only the 10 Fashion-MNIST classes are known, so unrelated objects are still assigned
one of them. It works best on a single item against a plain background.

## Live Deployment

The app is live and hosted on Streamlit Community Cloud:
🔗 **[https://fashion-classifier-5esprsdq2sikwhbijfuzsf.streamlit.app/](https://fashion-classifier-5esprsdq2sikwhbijfuzsf.streamlit.app/)**
