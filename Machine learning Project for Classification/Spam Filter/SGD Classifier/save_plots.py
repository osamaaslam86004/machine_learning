import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def plot_precision_recall_vs_threshold(y_true, y_scores):
    """
    Plots precision and recall as functions of the decision threshold.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_scores (array-like): Decision scores from the classifier.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="best")
    plt.title("Precision and Recall vs. Threshold")
    plt.grid(True)
    plt.savefig(
        "sgclassifier_roc_curve_precision_recall_vs_threshold.png"
    )  # Save the plot
    plt.show()


def plot_precision_vs_recall(precisions, recalls):
    """
    Plots precision against recall.

    Args:
        precisions (array-like): Precision values.
        recalls (array-like): Recall values.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.title("Precision vs. Recall")
    plt.grid(True)
    plt.savefig("sgclassifier_roc_curve_precision_vs_recall.png")  # Save the plot
    plt.show()


def plot_roc_curve(fpr, tpr, auc):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr (array-like): False Positive Rate values.
        tpr (array-like): True Positive Rate values.
        auc (float): Area Under the Curve (AUC) score.
    """

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("sgclassifier_roc_curve.png")
    plt.show()
