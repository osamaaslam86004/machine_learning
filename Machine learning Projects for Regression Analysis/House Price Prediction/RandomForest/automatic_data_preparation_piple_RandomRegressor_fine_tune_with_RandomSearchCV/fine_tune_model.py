from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV


def fine_tune_model(pipeline, X_train, y_train):
    # Define the hyperparameter distributions (you can widen the ranges if needed)
    param_distributions = {
        # You can use scipy.stats distributions (like randint) for continuous ranges
        "regressor__n_estimators": randint(50, 300),
        "regressor__max_features": randint(1, 3),
        "regressor__max_depth": [None, 10, 20, 30],
        "regressor__min_samples_split": [2, 5, 10],
    }

    # RandomizedSearchCV allows randomized hyperparameter tuning
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,  # number of random combinations to try
        # Too small might miss the best config; too large could be slow.
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
        verbose=2,
        error_score="raise",
        n_jobs=-1,  # use all CPU cores
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    # ===== STEP 1: Extract feature names from preprocessor and selector =====
    prep_step = best_model.named_steps["preparation"]
    selector_step = best_model.named_steps["feature_selector"]

    prep_step.fit(X_train)
    prep_features = prep_step.get_feature_names_out()

    final_feature_names = []
    for idx in [7, 9, selector_step.island_idx]:  # Ensure these indices are correct
        final_feature_names.append(prep_features[idx])

    # ===== STEP 2: Print feature importances =====
    feature_importances = best_model.named_steps["regressor"].feature_importances_

    print("\nTop Feature Importances AFTER Fine-Tuning:")
    for name, score in zip(final_feature_names, feature_importances):
        print(f"{name}: {score:.4f}")

    print("\nBest hyperparameters found:")
    print(random_search.best_params_)

    return best_model, random_search
