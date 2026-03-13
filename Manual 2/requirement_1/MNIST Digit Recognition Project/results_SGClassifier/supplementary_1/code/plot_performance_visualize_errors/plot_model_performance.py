import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_worst_errors_pca_applied(
    X, y_true, y_pred, decision_scores, n_images=10, type="training"
):
    """Plot data points of worst classification errors as 1D arrays."""
    wrong_idx = y_pred != y_true
    X_wrong = X[wrong_idx]
    y_true_wrong = y_true[wrong_idx]
    y_pred_wrong = y_pred[wrong_idx]
    scores_wrong = decision_scores[wrong_idx]
    confidences = np.max(scores_wrong, axis=1)
    sorted_idx = np.argsort(-confidences)
    top_wrong_idx = sorted_idx[:n_images]

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(top_wrong_idx):
        plt.subplot(2, 5, i + 1)
        plt.plot(X_wrong[idx])  # Plot as 1D array
        plt.title(f"Pred: {y_pred_wrong[idx]}\nTrue: {y_true_wrong[idx]}")
        plt.xlabel("Feature Index")  # Add x-axis label
        plt.ylabel("Feature Value")  # Add y-axis label
        plt.savefig(f"sgd_{type}_worst_misclassified_{i+1}.png")
        plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, type, model="sgd"):
    """
    Plots a confusion matrix in a human-readable format with counts.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(f"{model}_{type}__OvR_confusion_matrix.png")  # Saves the figure
    plt.show()
