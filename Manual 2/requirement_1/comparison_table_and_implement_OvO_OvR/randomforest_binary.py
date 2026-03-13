# # MNIST Classification with Random Forest Classifiers
#
# This code implements and evaluates RandomForestClassifier models
# on the MNIST dataset, for both binary (5-detector) and multiclass (0-9) tasks,
# drawing from examples and discussions in the provided source material.


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


def train_evaluate_random_forest_binary_classifier(X_train, X_test, y_train, y_test):
    """
    1. Training a RandomForest classifier to detect the digit 5.
    2. Evaluating using 3-fold cross-validation with accuracy, precision, recall, and F1 scores as the scoring metric.
    3. print confusion_matrix
    4. plot Precision, Recall vs Threshold
    5. plot ROC curve to get Area under ROC-curve
    """

    # %%
    # Create target vectors for binary classification (5 vs not-5)
    y_train_5 = y_train == 5
    y_test_5 = y_test == 5

    # Instantiate SGDClassifier
    # random_state=42 is set for reproducible results as in source
    random_forest_clf = RandomForestClassifier(random_state=42)

    y_train_pred = cross_val_predict(random_forest_clf, X_train, y_train_5, cv=3)

    accuracy = accuracy_score(y_train_5, y_train_pred)
    print(
        f"RandomforestClassifier (Binary 5-detector) Cross-Validation Accuracy: {accuracy}"
    )

    # The confusion matrix gives you a lot of information, but sometimes you may prefer a
    # more concise metric like Precision.
    confusion__matrix = confusion_matrix(y_train_5, y_train_pred)
    print(
        f"RandomForestClassifier (Binary 5-detector) Confusion Matrix: {confusion__matrix}"
    )

    scores_precision = precision_score(y_train_5, y_train_pred)
    print(
        f"SGDClassifier (Binary 5-detector) Precision using precision_score: {scores_precision}"
    )

    scores_recall = recall_score(y_train_5, y_train_pred)
    print(
        f"SGDClassifier (Binary 5-detector) Recall using recall_score: {scores_recall}"
    )

    # The F1 score favors classifiers that have similar precision and recall. This is not always
    # what you want: in some contexts you mostly care about precision, and in other contexts
    # you really care about recall
    scores_f1 = f1_score(y_train_5, y_train_pred)
    print(f"SGDClassifier (Binary 5-detector) F1 Scores: {scores_f1}")

    # Now with these scores you can compute precision and recall for all possible thresholds
    # using the precision_recall_curve() function:
    """
    Why use precision_recall_curve?
    We want to find best Precision/Recall tradeoff. For this we need Precision, and Recall vs threshold.
    Scikit-Learn does not let you set the threshold directly.
    """
    y_probas = cross_val_predict(
        random_forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
    )[:, 1]
    print(
        f"RandomForestClassifier (Binary 5-detector) prediction Probabilities: {y_probas}"
    )

    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_probas)

    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim(
            [0, 1]
        )  # defines the range of values that will be displayed on the y-axis.
        plt.title("Precision and Recall vs Threshold")
        plt.savefig(
            f"Binary Classification ({random_forest_clf.__class__.__name__}) Precision and Recall vs Threshold.png"
        )
        plt.show()

    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    # Another way to select a good precision/recall tradeoff is to plot precision directly
    # against recall

    # How to do it?
    # 1. To be more precise you can search for the lowest threshold that gives you at least 90% precision
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]  # ~7816

    # 2. To make predictions (on the training set for now), instead of calling the classifier’s
    # predict() method, you can just run this code:
    y_train_pred_90 = y_probas >= threshold_90_precision

    # 3. Let’s check these predictions’ precision and recall:
    precision_score_8000_threshold = precision_score(y_train_5, y_train_pred_90)
    print(f"Precision score for threshold=8000: {precision_score_8000_threshold}")

    recall_score_8000_threshold = recall_score(y_train_5, y_train_pred_90)
    print(f"Recall score for threshold=8000: {recall_score_8000_threshold}")

    # To plot the ROC curve / Area under ROC-curve, you first need to compute the
    # TPR and FPR for various threshold values, using the roc_curve() function:

    # Step 1:
    fpr, tpr, thresholds = roc_curve(y_train_5, y_probas)
    # Then you can plot the FPR against the TPR using Matplotlib

    def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
        plt.plot([0, 1], [0, 1], "k--", label="dashed diagonal")  # dashed diagonal
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.legend(loc="lower right")
        plt.title("ROC Curve")
        plt.savefig(
            f"Binary Classification ({random_forest_clf.__class__.__name__}) ROC curve.png"
        )
        plt.show()

    # Step 2
    plot_roc_curve(fpr, tpr)
    area_under_roc_curve = roc_auc_score(y_train_5, y_probas)
    """
    Since the ROC curve is so similar to the precision/recall (or PR)
    curve, you may wonder how to decide which one to use. As a rule
    of thumb, you should prefer the PR curve whenever the positive
    class is rare or when you care more about the false positives than
    the false negatives, and the ROC curve otherwise
    """
    print(f"Area under ROC Curve: {area_under_roc_curve}")
