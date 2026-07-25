from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def train_model(X_train_transformed, y_train):
    """
    Trains an SVM classifier with a non-linear kernel.

    Args:
        X_train_transformed:  Transformed training data.
        y_train: Training labels (0 for ham, 1 for spam).

    Returns:
        clf: Trained SVC model.
    """

    param_grid = {
        "C": [0.1, 1, 10],  # Regularization strength
        "kernel": ["rbf", "linear"],  # Non-linear kernel (Radial Basis Function)
        "gamma": ["scale", "auto"],  # Kernel coefficient
    }

    # Initialize the SVC with specified parameters:
    # random_state: Controls the random seed for reproducibility.
    clf = SVC(random_state=42)

    grid_search = GridSearchCV(
        clf, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=2
    )  # 3-fold cross-validation
    grid_search.fit(X_train_transformed, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    return grid_search.best_estimator_  # Return the best model
