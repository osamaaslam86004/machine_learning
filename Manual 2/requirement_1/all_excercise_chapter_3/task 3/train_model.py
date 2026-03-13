from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_model(X_train_prepared, y_train):
    # Define the parameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy"
    )

    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train_prepared, y_train)

    # Print the best parameters and the best score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Get the best estimator
    best_model = grid_search.best_estimator_

    return best_model
