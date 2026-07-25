from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_model(X_train_transformed, y_train):
    """
    Trains a Logistic Regression classifier.

    Args:
        X_train_tfidf:  TF-IDF vectorized training data.
        y_train: Training labels (0 for ham, 1 for spam).

    Returns:
        clf: Trained LogisticRegression model.
    """

    param_grid = {
        "C": [0.1, 1, 10],  # Regularization strength
        "solver": ["liblinear", "lbfgs"],
        "max_iter": [100, 500, 1000],
    }

    # Initialize the LogisticRegression with specified parameters:
    # random_state: Controls the random seed for reproducibility.
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)

    grid_search = GridSearchCV(
        clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )  # 5-fold cross-validation
    grid_search.fit(X_train_transformed, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    return grid_search.best_estimator_  # Return the best model
