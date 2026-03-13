# # MNIST Classification with Random Forest Classifiers
#
# This code implements and evaluates RandomForestClassifier models
# on the MNIST dataset, for multiclass (0-9) tasks,
# Stategy: One vs Rest


import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (  # Various metrics are discussed
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict


def plot_roc_curve_all_classes(y_train, y_probas):
    # To plot the ROC curve / Area under ROC-curve, you first need to compute the
    # TPR and FPR for various threshold values, using the roc_curve() function:
    # Step 1 : Get probabilities y_probas

    # Step 2: Set up subplots
    fig, axs = plt.subplots(5, 2, figsize=(12, 20))
    axs = axs.ravel()

    # Step 3: Loop over classes
    roc_aucs = []

    for c in range(10):
        # Binary target: class `c` vs rest
        y_true_binary = (y_train == c).astype(int)
        y_score_binary = y_probas[:, c]

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        auc = roc_auc_score(y_true_binary, y_score_binary)
        roc_aucs.append(auc)

        # Plot
        axs[c].plot(fpr, tpr, label=f"Digit {c} (AUC = {auc:.4f})")
        axs[c].plot([0, 1], [0, 1], "k--", linewidth=0.5)
        axs[c].set_title(f"ROC Curve for Digit {c}")
        axs[c].set_xlabel("False Positive Rate")
        axs[c].set_ylabel("True Positive Rate")
        axs[c].legend(loc="lower right")
        axs[c].grid(True)

    # Step 4: Finalize plot
    plt.tight_layout()
    plt.savefig("RF_multiclass_plot_roc_curve_all_classes.png")
    plt.show()

    # Step 5: Print AUCs and average
    print("\nROC AUC per class:")
    for c, auc in enumerate(roc_aucs):
        print(f"Digit {c}: AUC = {auc:.4f}")

    average_auc = np.mean(roc_aucs)
    print(f"\nAverage ROC AUC across all classes: {average_auc:.4f}")


def plot_precision_recall_vs_threshold(y_train, y_probas):
    fig, axs = plt.subplots(5, 2, figsize=(12, 20))
    axs = axs.ravel()

    for c in range(10):
        y_true_binary = (y_train == c).astype(int)
        y_score_binary = y_probas[:, c]
        precision, recall, thresholds = precision_recall_curve(
            y_true_binary, y_score_binary
        )

        axs[c].plot(thresholds, precision[:-1], "b--", label="Precision")
        axs[c].plot(thresholds, recall[:-1], "g-", label="Recall")
        axs[c].set_title(f"Digit {c}")
        axs[c].set_xlabel("Threshold")
        axs[c].set_ylabel("Score")
        axs[c].legend()
        axs[c].grid(True)

    plt.tight_layout()
    plt.savefig("RF_multiclass_precision_recall_vs_threshold_plot.png")
    plt.show()


def train_evaluate_random_forest_multiclass_classifier(
    X_train, X_test, y_train, y_test
):
    """
    1. Training a RandomForest classifier for Multinomial classification.
    2. Evaluating using 3-fold cross-validation with accuracy, precision, recall, and F1 scores as the scoring metric.
    3. print confusion_matrix
    4. plot Precision, Recall vs Threshold
    5. plot ROC curve to get Area under ROC-curve
    """

    # Instantiate SGDClassifier
    # random_state=42 is set for reproducible results as in source
    random_forest_clf = RandomForestClassifier(random_state=42)

    y_train_pred = cross_val_predict(random_forest_clf, X_train, y_train, cv=3)

    # The confusion matrix gives you a lot of information, but sometimes you may prefer a
    # more concise metric like Precision.
    confusion__matrix = confusion_matrix(y_train, y_train_pred)
    print(f"Confusion Matrix: {confusion__matrix}")

    accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Cross-Validation Accuracy: {accuracy}")

    scores_precision = precision_score(y_train, y_train_pred, average="micro")
    print(f"Precision using precision_score: {scores_precision}")

    scores_recall = recall_score(y_train, y_train_pred, average="micro")
    print(f"Recall using recall_score: {scores_recall}")

    # The F1 score favors classifiers that have similar precision and recall. This is not always
    # what you want: in some contexts you mostly care about precision, and in other contexts
    # you really care about recall
    scores_f1 = f1_score(y_train, y_train_pred, average="micro")
    print(f"F1 Scores: {scores_f1}")

    # Now with these probabilities you can compute precision and recall for all possible thresholds
    # using the precision_recall_curve() function:
    y_probas = cross_val_predict(
        random_forest_clf, X_train, y_train, cv=3, method="predict_proba"
    )
    print(f"\nprediction Probabilities: {y_probas}")

    """
    Why use precision_recall_curve?
    We want to find best Precision/Recall tradeoff. For this we need Precision, and Recall vs threshold.
    Scikit-Learn does not let you set the threshold directly.
    """
    print(f"\nPlotting Precision, Recall vs Threshold......... ")
    plot_precision_recall_vs_threshold(y_train, y_probas)

    """
    Since the ROC curve is so similar to the precision/recall (or PR)
    curve, you may wonder how to decide which one to use. As a rule
    of thumb, you should prefer the PR curve whenever the positive
    class is rare or when you care more about the false positives than
    the false negatives, and the ROC curve otherwise
    """
    print(f"\nPlotting ROC curve......... ")
    plot_roc_curve_all_classes(y_train, y_probas)
