# Required for hyperparameter tuning
import pandas as pd
from sklearn.model_selection import GridSearchCV

param_grid = {
    # Number of trees in the forest.  More trees generally lead to better performance but increase training time.
    "regressor__n_estimators": [50, 100, 150, 200],
    # The number of features to consider when looking for the best split.  Controls the diversity of trees in the forest
    "regressor__max_features": [1, 2],
}


def fine_tune_model(pipeline, X_train, y_train):
    """
    Fine-Tune Model Using Randomized Search:
    This code demonstrates using `GridSearchCV` to find better hyperparameters for the `RandomForestRegressor`.
    """

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        verbose=2,
        error_score="raise",  # to raise errors during fitting
    )

    # Run grid search
    grid_search = grid_search.fit(X_train, y_train)

    # Get the best model from the search
    fine_tuned_model = grid_search.best_estimator_

    # ===== STEP 1: Re-run only the `preparation` and `feature_selector` parts on training data to get correct feature names =====
    prep_step = fine_tuned_model.named_steps["preparation"]
    selector_step = fine_tuned_model.named_steps["feature_selector"]

    # Fit `preparation` again on X_train to extract feature names after transformation
    prep_step.fit(X_train)
    prep_features = prep_step.get_feature_names_out()

    # Feature selector selects 3 features: median_income, population_per_household, and island indicator
    # The selector's output order is: [7, 9, island_idx]
    final_feature_names = []
    for idx in [7, 9, selector_step.island_idx]:
        final_feature_names.append(prep_features[idx])

    # ===== STEP 2: Get feature importances from regressor =====
    feature_importances = fine_tuned_model.named_steps["regressor"].feature_importances_

    print("\nTop Feature Importances After Fine-Tuning:")
    for name, score in zip(final_feature_names, feature_importances):
        print(f"{name}: {score:.4f}")

    print("\nBest hyperparameters found:")
    print(grid_search.best_params_)

    return fine_tuned_model, grid_search
