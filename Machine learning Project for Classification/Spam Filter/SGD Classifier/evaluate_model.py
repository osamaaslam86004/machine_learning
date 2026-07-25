import matplotlib.pyplot as plt  # For plotting the ROC curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from save_plots import (
    plot_precision_recall_vs_threshold,
    plot_precision_vs_recall,
    plot_roc_curve,
)


def evaluate_model(clf, X_test_transformed, y_test):
    """
    Evaluates the trained classifier, including accuracy, classification report,
    ROC curve, and AUC score.

    Args:
        clf: Trained classifier.
        X_test_transformed: Transformed test data.
        y_test: True labels for the test data.

    Returns:
        y_pred: Predicted labels (0 or 1).
    """
    # Get decision scores
    y_scores = clf.decision_function(X_test_transformed)

    # Plot precision and recall vs. threshold curve
    plot_precision_recall_vs_threshold(y_test, y_scores)

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    plot_precision_vs_recall(precisions, recalls)

    # Predictions for accuracy and classification report (using a threshold of 0,
    # which is the default for SGDClassifier). This line can also be removed since
    # the focus is ROC and AUC
    y_pred = clf.predict(X_test_transformed)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # Calculate AUC
    auc = roc_auc_score(y_test, y_scores)
    print("\nAUC:", auc)

    # Plot ROC curve
    plot_roc_curve(fpr, tpr, auc)

    return y_pred  # Or return y_scores if you want scores instead of hard predictions
