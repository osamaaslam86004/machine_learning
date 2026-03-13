# # MNIST Classification with SGD multi-class Classifiers with One vs Rest strategy
#
# This code implements and evaluates SGDClassifier and RandomForestClassifier models
# OvA and OvO strategy with SGClassifier
# ROC curve

import joblib
import numpy as np
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


def train_sg_classifier_OvR_strategy(X_train, y_train, type="train"):
    """Trains an SGDClassifier using the One-vs-Rest (OvR) strategy with hyperparameter tuning via GridSearchCV.

    This function trains a multiclass SGDClassifier for digit classification (0-9) using the OvR strategy.
    It performs the following steps:
        1. Initializes an SGDClassifier with a hinge loss and a specified random state.
        2. Wraps the SGDClassifier with a OneVsRestClassifier to implement the OvR strategy.
        3. Scales the training data using StandardScaler.
        4. Defines a parameter grid for GridSearchCV to optimize hyperparameters such as
           regularization strength (alpha), penalty type (l1, l2, elasticnet), and maximum iterations.
        5. Performs GridSearchCV to find the best hyperparameter combination based on accuracy.
        6. Prints the best hyperparameters found.
        7. Extracts the best estimator (model) from the GridSearchCV results.
        8. Saves the best model to a file using joblib.
        9. Calculates the decision function scores for the training data using the best model.

    Args:
        X_train (numpy.ndarray): The PCA applied training data features.
        y_train (numpy.ndarray): The training data labels.
        type (str, optional): A string identifier for the training type (e.g., "train", "fine-tune"). Defaults to "train".

    Returns:
        tuple: A tuple containing:
            - best_ovr_clf (sklearn.multiclass.OneVsRestClassifier): The best trained OvR model.
            - X_train_scaled (numpy.ndarray): The scaled PCA applied training data.
            - decision_scores (numpy.ndarray): The decision function scores of the best model on the scaled PCA applied training data.
    """

    # Instantiate SGDClassifier
    sgd_clf = SGDClassifier(random_state=42, loss="hinge")
    ovr_clf = OneVsRestClassifier(sgd_clf)  # Apply OvR strategy

    # Scaling is recommended
    scaler = StandardScaler()
    # The StandardScaler needs to be fitted to the training data to learn the mean and standard deviation of each feature. This fitted scaler is then used to transform both the training and testing
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

    joblib.dump(scaler, f"scaler_{type}.joblib")
    print(f"Scaler saved to 'scaler_{type}.joblib'")

    # Define parameter grid for GridSearchCV
    param_grid = {
        "estimator__alpha": [0.0001, 0.001, 0.01],  # Regularization strength
        "estimator__penalty": ["l1", "l2", "elasticnet"],  # Regularization type
        "estimator__max_iter": [2000, 3000, 5000],
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(ovr_clf, param_grid, cv=3, scoring="accuracy", verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    # Print best hyperparameters
    print("Best hyperparameters found by GridSearchCV:")
    print(grid_search.best_params_)

    # Get the best estimator from GridSearchCV
    best_ovr_clf = grid_search.best_estimator_  # best model
    best_estimators = best_ovr_clf.estimators_  # list of trained model

    decision_scores = best_ovr_clf.decision_function(X_train_scaled)

    # Save the best model to a file
    joblib.dump(best_ovr_clf, f"best_sgd_ovr_{type}_model.joblib")
    print(f"Best model saved to 'best_sgd_ovr_{type}_model.joblib'")

    return best_ovr_clf, X_train_scaled, decision_scores


def evaluate_sg_classifier_one_vs_one_strategy(
    sgd_clf, X_train_scaled, y_train, type="train"
):
    """Evaluates a trained SGDClassifier using cross-validation and prints performance metrics.

    This function evaluates a multiclass SGDClassifier (trained with a One-vs-Rest strategy)
    on the training data using cross-validation. It calculates and prints the accuracy,
    precision, recall, F1 score, and confusion matrix.  It also generates and saves a
    visualization of the confusion matrix.

    Args:
        sgd_clf (sklearn.multiclass.OneVsRestClassifier): The trained SGDClassifier model.
        X_train_scaled (numpy.ndarray): The scaled PCA applied training data features.
        y_train (numpy.ndarray): The training data labels.
        type (str, optional):  A string identifier for the evaluation type (e.g., "train", "fine-tune").
                              This is used in the filename for saving the confusion matrix plot.
                              Defaults to "train".

    Returns:
        numpy.ndarray: The predicted labels for the PCA applied training data obtained via cross-validation.
    """

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
    plot_confusion_matrix(y_train, y_train_pred, classes, type)

    #  Add the classification report
    print(f"\nClassification Report for {type}:")
    print(classification_report(y_train, y_train_pred))

    return y_train_pred
