import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


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
    knn_clf = KNeighborsClassifier(algorithm="brute", n_jobs=-1)

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        knn_clf, param_grid, cv=3, scoring="accuracy", verbose=2, n_jobs=-1
    )  # Increased verbosity for better feedback
    grid_search.fit(X_train, y_train)

    # Get the best estimator and its parameters
    best_knn_clf = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\nBest parameters: {best_params}")

    return best_knn_clf


def train_model_with_pca(X_train, y_train):
    """
    1. Apply PCA to retain 95% variance
    2. Train the KNN-model with GridSearchCV using a pipeline
    3. Return the best model
    """

    # Create pipeline: PCA + KNN
    pipeline = Pipeline(
        [
            ("pca", PCA(n_components=0.95)),
            ("knn", KNeighborsClassifier(algorithm="kd_tree", n_jobs=-1)),
        ]
    )

    # Define GridSearch parameter grid for the KNN part of the pipeline
    param_grid = {
        "knn__n_neighbors": [3, 4, 5],  # 'knn__' is important: refers to step name
        "knn__weights": ["uniform", "distance"],
    }

    # Run GridSearchCV on the pipeline
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="accuracy",
        verbose=2,
        n_jobs=1,  # Limit to avoid killing workers
    )
    grid_search = grid_search.fit(X_train, y_train)

    # Get best estimator
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\n Best parameters: {best_params}")
    print(f"PCA selected dimensions: {best_model.named_steps['pca'].n_components_}")

    return best_model
