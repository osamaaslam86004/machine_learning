# Required for hyperparameter tuning
import numpy as np
from scipy.stats import expon, reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR


def fine_tune_model(housing_prepared, housing_labels):
    """
    Fine-Tune Model Using Randomized Search for RandomForest or SVR.
    """

    # Create new UNFITTED estimator with base parameters
    base_model = SVR(kernel="rbf")

    param_distributions = {
        "kernel": ["rbf"],
        "C": np.logspace(-2, 3, 20),  # from 0.01 to 1000
        "gamma": np.logspace(-4, 1, 20),  # from 0.0001 to 10
    }

    scoring = "neg_mean_squared_error"  # Or try "neg_root_mean_squared_error"

    rnd_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=30,
        cv=5,
        scoring=scoring,
        random_state=42,
        verbose=2,
        n_jobs=-1,
    )

    rnd_search.fit(housing_prepared, housing_labels)

    final_model = rnd_search.best_estimator_

    return final_model, rnd_search
