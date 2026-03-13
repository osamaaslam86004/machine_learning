# # MNIST Classification with SGD multi-class Classifiers with One vs Rest strategy
#
# This code implements and evaluates SGDClassifier and RandomForestClassifier models
# OvA and OvO strategy with SGClassifier
# ROC curve

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from plot_performance_visualize_errors.plot_model_performance import (
    plot_confusion_matrix,
)


def train_random_forest_OvR_strategy(X_train, y_train, type="train"):
    """Trains a RandomForestClassifier using the One-vs-Rest (OvR) strategy with hyperparameter tuning via GridSearchCV.

    This function trains a multiclass RandomForestClassifier for digit classification (0-9) using the OvR strategy.
    It performs the following steps:
        1. Initializes a RandomForestClassifier with a specified random state.
        2. Wraps the RandomForestClassifier with a OneVsRestClassifier to implement the OvR strategy.
        3. Defines a parameter grid for GridSearchCV to optimize hyperparameters such as
           number of estimators and max depth.
        4. Performs GridSearchCV to find the best hyperparameter combination based on accuracy.
        5. Prints the best hyperparameters found.
        6. Extracts the best estimator (model) from the GridSearchCV results.
        7. Saves the best model to a file using joblib.
        8. Calculates the decision function scores for the training data using the best model.

    Args:
        X_train (numpy.ndarray): The PCA applied training data features.
        y_train (numpy.ndarray): The training data labels.
        type (str, optional): A string identifier for the training type (e.g., "train", "fine-tune"). Defaults to "train".

    Returns:
        tuple: A tuple containing:
            - best_ovr_clf (sklearn.multiclass.OneVsRestClassifier): The best trained OvR model.
            - X_train (numpy.ndarray): The training data features (no scaling for RandomForest).
            - decision_scores (numpy.ndarray): The decision function scores of the best model on the training data.
    """

    # Instantiate RandomForestClassifier
    rf_clf = RandomForestClassifier(random_state=42)
    ovr_clf = OneVsRestClassifier(rf_clf)  # Apply OvR strategy

    # Define parameter grid for GridSearchCV
    param_grid = {
        "estimator__n_estimators": [3, 4],  # Number of trees in the forest
        "estimator__max_depth": [None, 10, 20],  # Maximum depth of the trees
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(ovr_clf, param_grid, cv=3, scoring="f1_micro", verbose=2)
    grid_search.fit(X_train, y_train)

    # Print best hyperparameters
    print("Best hyperparameters found by GridSearchCV:")
    print(grid_search.best_params_)

    # Get the best estimator from GridSearchCV
    best_ovr_clf = grid_search.best_estimator_  # best model

    # RandomForest doesn't have decision_function, using predict_proba instead
    decision_scores = best_ovr_clf.predict_proba(X_train)

    # Save the best model to a file
    joblib.dump(best_ovr_clf, f"best_rf_ovr_{type}_model.joblib")
    print(f"Best model saved to 'best_rf_ovr_{type}_model.joblib'")

    return best_ovr_clf, X_train, decision_scores


def evaluate_random_forest_one_vs_one_strategy(rf_clf, X_train, y_train, type="train"):
    """Evaluates a trained RandomForestClassifier using cross-validation and prints performance metrics.

    This function evaluates a multiclass RandomForestClassifier (trained with a One-vs-Rest strategy)
    on the training data using cross-validation. It calculates and prints the accuracy,
    precision, recall, F1 score, and confusion matrix.  It also generates and saves a
    visualization of the confusion matrix.

    Args:
        rf_clf (sklearn.multiclass.OneVsRestClassifier): The trained RandomForestClassifier model.
        X_train (numpy.ndarray): The training data features.
        y_train (numpy.ndarray): The training data labels.
        type (str, optional):  A string identifier for the evaluation type (e.g., "train", "fine-tune").
                              This is used in the filename for saving the confusion matrix plot.
                              Defaults to "train".

    Returns:
        numpy.ndarray: The predicted labels for the training data obtained via cross-validation.
    """

    model = "Random Forest"

    # Use cross-validation predictions on the training set for evaluation
    y_train_pred = cross_val_predict(rf_clf, X_train, y_train, cv=3)

    print(f"Accuracy: {accuracy_score(y_train, y_train_pred)}")
    print(f"Precision: {precision_score(y_train, y_train_pred, average='macro')}")
    print(f"Recall: {recall_score(y_train, y_train_pred, average='macro')}")
    print(f"F1 Score: {f1_score(y_train, y_train_pred, average='macro')}")

    print_confusion_matrix = confusion_matrix(y_train, y_train_pred)
    print(f"Confusion Matrix: \n{print_confusion_matrix}")

    classes = np.arange(10)  # Define classes here
    plot_confusion_matrix(y_train, y_train_pred, classes, model, type)

    #  Add the classification report
    print(f"\nClassification Report for {type}:")
    print(classification_report(y_train, y_train_pred))

    return y_train_pred
