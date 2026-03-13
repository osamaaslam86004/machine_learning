import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import GridSearchCV


def select_important_features(X_train_prepared, y_train):
    # Step 1: Train the initial model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_prepared, y_train)

    # Step 2: Select important features based on model's importances
    selector = SelectFromModel(model, prefit=True, threshold="median")

    # Step 3: Transform datasets to only include selected features
    X_train_selected = selector.transform(X_train_prepared)

    return X_train_selected, selector


def fine_tune_model(X_train_selected, y_train):
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy"
    )
    grid_search.fit(X_train_selected, y_train)

    print("Best parameters:", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    return best_model


# def recursive_feature_elimination_with_gridsearch(X, y, num_features=10):
#     # Step 1: Feature Selection using RFE
#     base_model = RandomForestClassifier(random_state=42)
#     rfe = RFE(estimator=base_model, n_features_to_select=num_features)
#     rfe.fit(X, y)

#     # Select the important features
#     if isinstance(X, pd.DataFrame):
#         selected_features = X.columns[rfe.support_]
#         X_selected = X[selected_features]
#         print("Selected Features:", selected_features.tolist())
#     else:
#         X_selected = X[:, rfe.support_]

#     # Step 2: Define GridSearchCV on selected features
#     param_grid = {
#         "n_estimators": [50, 100, 200],
#         "max_depth": [None, 10, 20],
#         "min_samples_split": [2, 5],
#         "min_samples_leaf": [1, 2],
#     }

#     grid_search = GridSearchCV(
#         RandomForestClassifier(random_state=42),
#         param_grid,
#         cv=5,
#         scoring="accuracy",
#         n_jobs=-1,
#     )

#     grid_search.fit(X_selected, y)

#     # Step 3: Evaluate best model
#     best_model = grid_search.best_estimator_
#     print("\nBest Parameters from GridSearchCV:", grid_search.best_params_)
#     print("Best Cross-validation Score:", grid_search.best_score_)

#     return best_model, selected_features
