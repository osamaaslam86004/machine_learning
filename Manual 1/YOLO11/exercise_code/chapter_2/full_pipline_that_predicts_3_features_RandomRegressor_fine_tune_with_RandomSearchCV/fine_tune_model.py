import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "regressor__n_estimators": randint(low=1, high=200),
    "regressor__max_features": randint(low=1, high=3),
}


def fine_tune_model(pipeline, X_train, y_train):
    """
    Fine-Tune Model Using Randomized Search:
    This code demonstrates using `RandomizedSearchCV` to find better hyperparameters for the `RandomForestRegressor`.
    It prints feature importances based on the selected features in the pipeline.
    """

    rnd_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
        verbose=2,
    )

    rnd_search = rnd_search.fit(X_train, y_train)
    fine_tuned_model = rnd_search.best_estimator_

    # Extract feature names *after* preprocessing
    feature_names = pipeline.named_steps["preparation"].get_feature_names_out()

    # Get selected feature names from the feature selector
    selected_indices = pipeline.named_steps[
        "feature_selector"
    ].island_idx  # Assuming island_idx holds the index
    selected_feature_names = [
        feature_names[idx] for idx in [7, 9, selected_indices]
    ]  # median_income and population_per_household indices are hardcoded

    # Get feature importances from the regressor
    feature_importances = fine_tuned_model.named_steps["regressor"].feature_importances_

    print("\nFeature Importances AFTER Tuning:")
    for name, score in zip(selected_feature_names, feature_importances):
        print(f"{name}: {score:.4f}")

    return fine_tuned_model, rnd_search
