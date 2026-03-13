import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes):
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
    plt.savefig("train_KNN_confusion_matrix.png")  # Saves the figure
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a trained model on a test dataset.

    This function calculates and prints several evaluation metrics, including:
        - Accuracy: The overall accuracy of the model's predictions.
        - Classification Report:  A detailed report including precision, recall, F1-score, and support for each class.
        - Confusion Matrix: A matrix showing the counts of true vs. predicted labels for each class.
        - Prediction Time: The time taken for the model to make predictions on the test set.

    It also generates and saves a visual representation of the confusion matrix as a PNG image.

    Args:
        model: The trained scikit-learn model to evaluate.
        X_test: The test dataset features.
        y_test: The true labels for the test dataset.

    Returns:
        None. The function prints the evaluation metrics and saves the confusion matrix plot to a file.
    """

    # Time the prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    # 5. Evaluation metrics
    print("\n Accuracy:", accuracy_score(y_test, y_pred))
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n Prediction Time: %.2f seconds" % (end_time - start_time))

    classes = np.arange(10)  # Define classes here
    plot_confusion_matrix(y_test, y_pred, classes)
