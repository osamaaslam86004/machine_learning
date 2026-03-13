import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def train_model(X_train, y_train):
    """
    1. Train the KNN-model using with best hyper-parameters (GridsearchCV)
    2. return best model
    """

    # Define the parameter grid
    param_grid = {
        "n_neighbors": [3, 4, 5],  # You can expand this range
        "weights": ["uniform", "distance"],
    }

    # Initialize the KNeighborsClassifier
    knn_clf = KNeighborsClassifier()

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        knn_clf, param_grid, cv=3, scoring="accuracy", verbose=2
    )  # Increased verbosity for better feedback
    grid_search.fit(X_train, y_train)

    # Get the best estimator and its parameters
    best_knn_clf = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\nBest parameters: {best_params}")

    return best_knn_clf
