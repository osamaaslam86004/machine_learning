from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


def train_model(X_train_transformed, y_train):
    """
    Trains a linear classifier using Stochastic Gradient Descent (SGD).

    Args:
        X_train_tfidf:  TF-IDF vectorized training data.
        y_train: Training labels (0 for ham, 1 for spam).

    Returns:
        clf: Trained SGDClassifier model.
    """

    param_grid = {
        # loss: Specifies the loss function (hinge for Support Vector Machine).
        # penalty: Specifies the regularization penalty (L2 regularization).
        "loss": ["hinge", "log_loss", "modified_huber"],
        "penalty": ["l1", "l2", "elasticnet"],
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1],  # Regularization strength
    }

    # Initialize the SGDClassifier with specified parameters:
    # random_state: Controls the random seed for reproducibility.
    # max_iter: Maximum number of iterations for the solver.
    # tol: Tolerance for the stopping criterion.
    clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)

    grid_search = GridSearchCV(
        clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )  # 5-fold cross-validation
    grid_search.fit(X_train_transformed, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    return grid_search.best_estimator_  # Return the best model
