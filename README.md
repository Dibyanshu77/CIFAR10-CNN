# CIFAR-10 Image Classification using CNN

## Problem
Classify 32x32 color images into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using a Convolutional Neural Network.

## Dataset
CIFAR-10 — 60,000 images (50,000 train / 10,000 test), loaded directly via `tf.keras.datasets.cifar10` (no manual download needed).

## Approach
- Normalized pixel values to the 0–1 range.
- Applied data augmentation (rotation, width/height shift, horizontal flip) to reduce overfitting.
- Built a CNN with 3 convolutional blocks (Conv2D + BatchNorm + MaxPooling + Dropout), increasing filter depth (32 → 64 → 128).
- Used dropout (0.25–0.5) and batch normalization throughout to stabilize training and reduce overfitting.
- Trained with Adam optimizer and early stopping on validation loss.

## My Contribution
- Implemented and tuned the CNN architecture (number of conv blocks, filter sizes, dropout rates).
- Added batch normalization on top of the base architecture to improve training stability.
- Ran experiments on augmentation settings and dropout rates to reduce the train/validation accuracy gap.
- Evaluated the model using a full classification report and confusion matrix, not just accuracy, to check per-class performance.

## Results
- **Test accuracy: 81.13%** after 30 epochs
- Classification report shows strong performance on automobile, ship, truck, and frog classes; cat/bird/dog show more confusion due to visual similarity (see confusion_matrix.png)
- See `training_curves.png` for accuracy/loss trends across epochs

## How to Run
1. Open `cifar.py` in Google Colab (enable GPU: Runtime > Change runtime type > GPU).
2. Run all cells — the dataset downloads automatically.
3. Outputs: trained model, confusion matrix, and training curve plots.

## Tech Stack
Python, TensorFlow/Keras, NumPy, Matplotlib, Seaborn, Scikit-learn
