# # MNIST Classification with SGD multi-class Classifiers with One vs One strategy
#
# This code implements and evaluates SGDClassifier model
# OvA and OvO strategy with SGClassifier
# ROC curve


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler


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
    plt.savefig("sg_multi_class_OvO_confusion_matrix.png")  # Saves the figure
    plt.show()


def evaluate_sg_classifier_one_vs_one_strategy(X_train, y_train, X_test, y_test):
    """Evaluates a classifier and prints relevant metrics."""

    # Instantiate SGDClassifier
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf = OneVsOneClassifier(sgd_clf)  # Apply OvR strategy

    # Scaling is recommended
    """
    scaler = StandardScaler(): This line initializes a StandardScaler object. The StandardScaler transforms the data to have zero mean and unit variance. Essentially, it standardizes each feature (pixel value in this case) by subtracting the mean and dividing by the standard deviation of that feature across all training samples.

    X_train.astype(np.float64): This converts the data type of X_train to np.float64. Scaling often works best with floating-point numbers to avoid potential issues with integer arithmetic.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

    # Use cross-validation predictions on the training set for evaluation
    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

    print(f"Accuracy: {accuracy_score(y_train, y_train_pred)}")

    # possibble values None, weighted, micro
    # This is because the output of wrapped classifiers is multiclass
    print(f"Precision: {precision_score(y_train, y_train_pred, average='macro')}")

    # This is because the output of wrapped classifiers is multiclass
    print(f"Recall: {recall_score(y_train, y_train_pred, average='macro')}")

    # This is because the output of wrapped classifiers is multiclass
    print(f"F1 Score: {f1_score(y_train, y_train_pred, average='macro')}")

    print_confusion_matrix = confusion_matrix(y_train, y_train_pred)
    print(f"Confusion Matrix: \n{print_confusion_matrix}")

    classes = np.arange(10)  # Define classes here
    plot_confusion_matrix(y_train, y_train_pred, classes)
