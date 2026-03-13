from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def fine_tune_model(model, X_train, y_train):
    """
    Fine-tunes a KNN model by performing a grid search over a narrower
    range of hyperparameters centered around the model's current parameters.

    Args:
        model: The KNN model to fine-tune.
        X_train: The training data.
        y_train: The training labels.

    Returns:
        The fine-tuned KNN model.
    """

    # Get the current parameters of the model
    current_params = model.get_params()
    n_neighbors = current_params["n_neighbors"]
    weights = current_params["weights"]

    # Define the parameter grid for fine-tuning
    param_grid = {
        "n_neighbors": [
            max(1, n_neighbors - 1),
            n_neighbors,
            min(n_neighbors + 1, 10),
        ],  # Adjust range as needed
        "weights": [weights],  # Keep weights the same
    }

    # 'brute' is default and Slowest — exact distance on full dataset
    #
    # auto	Lets scikit-learn decide
    # kd_tree	Works well for low–medium dimensions (e.g., < 30)
    # ball_tree	Similar to kd_tree, slightly better for high-dimensional data
    knn_clf = KNeighborsClassifier(algorithm="brute", n_jobs=-1)

    # Perform GridSearchCV for fine-tuning
    grid_search = GridSearchCV(
        knn_clf,  # Use a new classifier instance
        param_grid,
        cv=3,
        scoring="accuracy",
        verbose=2,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    # Get the best estimator from fine-tuning
    best_fine_tune_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\nBest parameters after fine-tuning: {best_params}")

    return best_fine_tune_model


def fine_tune_model_with_pca(model, X_train, y_train):
    """
    Fine-tunes a PCA+KNN pipeline by performing GridSearchCV on a narrow
    range of hyperparameters centered around the current KNN parameters.

    Args:
        model: The trained PCA+KNN pipeline model.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        A fine-tuned PCA+KNN pipeline model.
    """

    # Extract current KNN step parameters from pipeline
    current_knn = model.named_steps["knn"]
    n_neighbors = current_knn.get_params()["n_neighbors"]
    weights = current_knn.get_params()["weights"]

    # Define the parameter grid for fine-tuning (KNN only)
    param_grid = {
        "knn__n_neighbors": [
            max(1, n_neighbors - 1),
            n_neighbors,
            min(n_neighbors + 1, 10),
        ],
        "knn__weights": [weights],
    }

    # Rebuild the pipeline with the same PCA config
    pca_step = model.named_steps["pca"]
    pipeline = Pipeline(
        [
            ("pca", PCA(n_components=pca_step.n_components_)),
            ("knn", KNeighborsClassifier(algorithm="kd_tree", n_jobs=-1)),
        ]
    )

    # Perform fine-tuning
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="accuracy",
        verbose=2,
        n_jobs=1,  # Limit to avoid killing workers
    )  # n_jobs=-1 is not ised here to prevent (Even in GPU runtime) TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated... is caused by excessive memory usage,

    grid_search = grid_search.fit(X_train, y_train)

    best_fine_tune_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\n Best parameters after fine-tuning: {best_params}")
    print(
        f" PCA retained {best_fine_tune_model.named_steps['pca'].n_components_} components"
    )

    return best_fine_tune_model
