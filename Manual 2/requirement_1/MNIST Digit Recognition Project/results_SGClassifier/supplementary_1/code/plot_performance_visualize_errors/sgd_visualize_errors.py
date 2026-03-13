import matplotlib.pyplot as plt
import numpy as np


def plot_worst_errors(
    X_original,
    y_true,
    y_pred,
    decision_scores,
    n_images=10,
    model="sgd",
    type="training",
):
    """Plots the original images of the most confidently misclassified samples.

    This function identifies the samples that a classifier misclassified with the
    highest confidence and visualizes their original, uncompressed images.  It helps
    in understanding the types of errors the model is making.

    Args:
        X_original (numpy.ndarray): The original, uncompressed feature data (not PCA reduced),
            where each row represents a sample and each column a feature (pixel value).  Assumed
            to be in a format that can be reshaped into 28x28 images.
        y_true (numpy.ndarray): The true labels for the samples.
        y_pred (numpy.ndarray): The predicted labels for the samples.
        decision_scores (numpy.ndarray): The decision function scores from the classifier.
            Shape should be (n_samples, n_classes), where higher scores indicate greater
            confidence in the prediction.
        n_images (int, optional): The number of worst misclassified images to display.
            Defaults to 10.
        type (str, optional): A string describing the type of data being visualized
            (e.g., "training", "validation").  This is used in the filename when saving
            the plot.  Defaults to "training".

    Returns:
        None: The function displays and saves a plot of the worst misclassified images.
    """

    wrong_idx = y_pred != y_true

    # Use original uncompressed image data
    X_wrong_original = X_original[wrong_idx]
    y_true_wrong = y_true[wrong_idx]
    y_pred_wrong = y_pred[wrong_idx]
    scores_wrong = decision_scores[wrong_idx]

    # Find wrong predictions with highest confidence
    confidences = np.max(scores_wrong, axis=1)
    sorted_idx = np.argsort(-confidences)
    top_wrong_idx = sorted_idx[:n_images]

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(top_wrong_idx):
        plt.subplot(2, 5, i + 1)
        image = X_wrong_original[idx].reshape(28, 28)
        plt.imshow(image, cmap="gray")
        plt.title(f"Pred: {y_pred_wrong[idx]}\nTrue: {y_true_wrong[idx]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{model}_{type}_worst_misclassified.png")
