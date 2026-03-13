from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


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

    # Perform GridSearchCV for fine-tuning
    grid_search = GridSearchCV(
        KNeighborsClassifier(),  # Use a new classifier instance
        param_grid,
        cv=3,
        scoring="accuracy",
        verbose=2,
    )
    grid_search.fit(X_train, y_train)

    # Get the best estimator from fine-tuning
    best_fine_tune_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\nBest parameters after fine-tuning: {best_params}")

    return best_fine_tune_model
